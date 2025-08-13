
# === Advanced Plant Disease Diagnosis with Enhanced Accuracy ===

# Install required packages
!pip install -q opencv-python-headless pillow numpy matplotlib
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install -q transformers gradio scikit-learn

import io
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# --- Utility Functions ---

def variance_of_laplacian(image):
    """Measure focus quality."""
    try:
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 50.0  # Default reasonable focus score

def exposure_score(bgr_image):
    """Measure brightness."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.std(gray))

def gray_world_cc(image_bgr):
    """Apply white balance correction."""
    img = image_bgr.astype(np.float32)
    for i in range(3):
        if np.mean(img[:, :, i]) > 0:
            scale = np.mean(img[:, :, 1]) / np.mean(img[:, :, i])
            img[:, :, i] *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def detect_plant_content(pil_img):
    """Enhanced plant/leaf content detection using color analysis and edge detection."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Multiple green ranges to catch different types of vegetation
        green_ranges = [
            ([25, 30, 30], [85, 255, 255]),    # Primary green range
            ([35, 20, 20], [95, 255, 255]),    # Extended green range
            ([15, 25, 25], [100, 255, 255])   # Broader range for yellowing/browning leaves
        ]

        total_green_pixels = 0
        total_pixels = bgr.shape[0] * bgr.shape[1]

        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            # Ensure mask is single channel
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            total_green_pixels += cv2.countNonZero(mask)

        green_ratio = min(total_green_pixels / total_pixels, 1.0)

        # Check for leaf-like textures using edge detection
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Ensure edges is single channel
        if len(edges.shape) > 2:
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        edge_density = cv2.countNonZero(edges) / total_pixels

        # Enhanced detection criteria
        is_plant = (green_ratio > 0.1 and edge_density > 0.02) or green_ratio > 0.15

        return is_plant, green_ratio, edge_density

    except Exception as e:
        # Fallback: assume it might be a plant if we can't analyze properly
        return True, 0.1, 0.02

# --- Enhanced Disease Labels & Recognition ---

# Comprehensive disease database with scientific names and detailed symptoms
disease_database = {
    "healthy_leaf": {
        "common_name": "Healthy Leaf",
        "scientific_name": "Normal Plant Tissue",
        "symptoms": ["Vibrant green color", "No discoloration", "Smooth texture", "No spots or lesions"],
        "prompts": [
            "a perfectly healthy vibrant green leaf with no disease symptoms",
            "normal healthy plant leaf with uniform green coloration",
            "pristine leaf tissue with no spots discoloration or damage",
            "healthy foliage showing optimal plant health"
        ]
    },
    "powdery_mildew": {
        "common_name": "Powdery Mildew",
        "scientific_name": "Erysiphales species",
        "symptoms": ["White powdery coating", "Flour-like appearance", "Upper leaf surface affected", "Yellowing underneath powder"],
        "prompts": [
            "leaf infected with white powdery mildew fungus coating the surface",
            "plant leaf covered in characteristic white powdery fungal growth",
            "powdery white flour-like fungal infection on leaf surface",
            "white dusty mildew covering plant leaves"
        ]
    },
    "leaf_rust": {
        "common_name": "Leaf Rust",
        "scientific_name": "Puccinia species",
        "symptoms": ["Orange-red pustules", "Rusty colored spots", "Raised bumps on leaf surface", "Yellow halos around pustules"],
        "prompts": [
            "plant leaf with orange rust fungus pustules and spores",
            "leaf showing rusty orange colored fungal infection spots",
            "orange-red rust pustules erupting from leaf surface",
            "characteristic rust disease with orange spore masses"
        ]
    },
    "leaf_scorch": {
        "common_name": "Leaf Scorch/Necrosis",
        "scientific_name": "Environmental Stress Response",
        "symptoms": ["Brown dry edges", "Necrotic tissue", "Crispy texture", "Progressive browning from margins"],
        "prompts": [
            "leaf with brown scorched edges from environmental stress",
            "necrotic brown patches and dead tissue on plant leaves",
            "leaf scorch showing brown crispy margins and dead zones",
            "environmental damage with browning and desiccation"
        ]
    },
    "bacterial_spot": {
        "common_name": "Bacterial Spot",
        "scientific_name": "Xanthomonas/Pseudomonas species",
        "symptoms": ["Dark water-soaked spots", "Yellow halos", "Angular lesions", "Leaf perforation in severe cases"],
        "prompts": [
            "leaf with bacterial spot disease showing dark lesions with yellow halos",
            "bacterial infection causing water-soaked spots on plant leaves",
            "dark angular bacterial lesions with characteristic yellow borders",
            "bacterial pathogen damage with necrotic spots and chlorosis"
        ]
    },
    "downy_mildew": {
        "common_name": "Downy Mildew",
        "scientific_name": "Peronospora/Plasmopara species",
        "symptoms": ["Yellow angular patches", "Fuzzy gray growth on underside", "Water-soaked appearance", "Systemic yellowing"],
        "prompts": [
            "leaf with downy mildew showing yellow patches and gray fuzzy growth underneath",
            "downy mildew infection with angular yellow lesions on leaf surface",
            "plant leaf with characteristic downy fungal growth on lower surface",
            "yellow angular spots with grayish downy sporulation beneath"
        ]
    },
    "black_spot": {
        "common_name": "Black Spot",
        "scientific_name": "Diplocarpon rosae",
        "symptoms": ["Circular black spots", "Yellow halos", "Premature leaf drop", "Feathered edges on spots"],
        "prompts": [
            "leaf with black spot fungal disease showing circular dark lesions",
            "black spot infection with characteristic circular black patches",
            "fungal black spot disease with yellow halos around dark centers",
            "circular black fungal lesions with feathered irregular edges"
        ]
    },
    "anthracnose": {
        "common_name": "Anthracnose",
        "scientific_name": "Colletotrichum species",
        "symptoms": ["Brown irregular spots", "Sunken lesions", "Pink spore masses", "Leaf distortion"],
        "prompts": [
            "leaf with anthracnose fungal disease showing brown sunken lesions",
            "anthracnose infection with irregular brown spots and pink spore masses",
            "fungal anthracnose causing sunken necrotic lesions on leaves",
            "brown irregular anthracnose lesions with characteristic sunken appearance"
        ]
    },
    "aphid_damage": {
        "common_name": "Aphid Damage",
        "scientific_name": "Aphidoidea species",
        "symptoms": ["Curled leaves", "Sticky honeydew", "Yellow stippling", "Stunted growth"],
        "prompts": [
            "leaf damage from aphid infestation showing curling and yellowing",
            "aphid feeding damage with characteristic leaf curling and distortion",
            "plant leaf affected by sap-sucking aphids with yellowing patterns",
            "aphid damage showing stunted growth and honeydew deposits"
        ]
    },
    "spider_mite_damage": {
        "common_name": "Spider Mite Damage",
        "scientific_name": "Tetranychidae species",
        "symptoms": ["Fine stippling", "Bronze coloration", "Webbing", "Leaf desiccation"],
        "prompts": [
            "leaf with spider mite damage showing fine stippling and bronzing",
            "spider mite feeding damage with characteristic yellow speckles",
            "plant leaf with mite damage showing fine webbing and stippled appearance",
            "spider mite infestation causing bronze coloration and fine webbing"
        ]
    },
    "thrip_damage": {
        "common_name": "Thrip Damage",
        "scientific_name": "Thysanoptera species",
        "symptoms": ["Silver streaks", "Black specks", "Scarred surface", "Curled leaf edges"],
        "prompts": [
            "leaf with thrip damage showing silver streaks and black specks",
            "thrip feeding damage with characteristic silvery scarring",
            "plant leaf damaged by thrips with streaky silver appearance",
            "thrip damage showing silvery feeding scars and black fecal spots"
        ]
    },
    "nutrient_deficiency": {
        "common_name": "Nutrient Deficiency",
        "scientific_name": "Nutritional Disorder",
        "symptoms": ["Interveinal chlorosis", "Purple/red coloration", "Stunted growth", "Poor leaf development"],
        "prompts": [
            "leaf showing nutrient deficiency with interveinal chlorosis",
            "nutritional deficiency causing yellowing between leaf veins",
            "plant leaf with nutrient deficiency symptoms and poor coloration",
            "mineral deficiency showing characteristic yellowing and purpling"
        ]
    }
}

# Create comprehensive label lists
labels = list(disease_database.keys())
all_prompts = []
prompt_to_label = {}

for label, data in disease_database.items():
    for prompt in data["prompts"]:
        all_prompts.append(prompt)
        prompt_to_label[prompt] = label

# --- Load Enhanced Model ---

print("Loading enhanced CLIP model for plant disease recognition...")

# Use more capable CLIP model for better accuracy
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cpu"  # CPU optimized
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"Model loaded successfully on {device}")

# --- Comprehensive Treatment Database ---

TREATMENTS = {
    "healthy_leaf": {
        "description": "Your plant appears healthy! Continue with preventive care.",
        "severity": "None",
        "immediate_actions": [
            "Continue regular care routine",
            "Monitor for early disease signs",
            "Maintain optimal growing conditions"
        ],
        "treatment_options": {
            "preventive": [
                "Apply balanced fertilizer monthly",
                "Ensure proper spacing for air circulation",
                "Water at soil level, avoid wetting leaves"
            ]
        },
        "organic_solutions": [
            "Compost tea application",
            "Neem oil as preventive spray (monthly)",
            "Beneficial microorganism inoculants"
        ],
        "timeline": "Maintain current care routine"
    },

    "powdery_mildew": {
        "description": "Fungal disease creating white powdery coating on leaves. Thrives in humid conditions with poor air circulation.",
        "severity": "Moderate",
        "immediate_actions": [
            "Remove affected leaves immediately",
            "Improve air circulation around plants",
            "Reduce humidity levels",
            "Stop overhead watering"
        ],
        "treatment_options": {
            "organic": [
                "Baking soda spray (1 tbsp + 1/2 tsp liquid soap per gallon)",
                "Milk solution (1 part milk to 2 parts water)",
                "Sulfur-based fungicide application"
            ],
            "chemical": [
                "Potassium bicarbonate fungicide",
                "Propiconazole-based treatments",
                "Myclobutanil applications"
            ]
        },
        "organic_solutions": [
            "Neem oil spray (apply in evening, weekly)",
            "Compost tea with beneficial microorganisms",
            "Hydrogen peroxide solution (1:10 with water)"
        ],
        "timeline": "Treatment should show results in 7-14 days"
    },

    "leaf_rust": {
        "description": "Fungal disease causing orange-red pustules. Spreads rapidly in warm, moist conditions.",
        "severity": "Moderate to High",
        "immediate_actions": [
            "Remove and destroy infected leaves",
            "Avoid working with wet plants",
            "Increase air circulation",
            "Apply protective fungicide"
        ],
        "treatment_options": {
            "organic": [
                "Copper sulfate spray (follow label directions)",
                "Bordeaux mixture application",
                "Sulfur dust or spray"
            ],
            "chemical": [
                "Tebuconazole-based fungicides",
                "Propiconazole treatments",
                "Chlorothalonil applications"
            ]
        },
        "organic_solutions": [
            "Neem oil with copper soap",
            "Garlic and chili pepper extract",
            "Bicarbonate of soda spray"
        ],
        "timeline": "Early treatment crucial - 10-21 days for control"
    },

    "leaf_scorch": {
        "description": "Environmental stress causing brown, crispy leaf edges. Often due to water stress, heat, or wind damage.",
        "severity": "Low to Moderate",
        "immediate_actions": [
            "Provide consistent deep watering",
            "Apply organic mulch around base",
            "Provide afternoon shade if possible",
            "Check for root problems"
        ],
        "treatment_options": {
            "cultural": [
                "Improve soil drainage if waterlogged",
                "Add organic matter to retain moisture",
                "Install windbreaks if wind damage suspected"
            ]
        },
        "organic_solutions": [
            "Seaweed extract foliar spray",
            "Compost mulch application",
            "Mycorrhizal inoculant for root health"
        ],
        "timeline": "Recovery depends on addressing underlying cause"
    },

    "bacterial_spot": {
        "description": "Bacterial infection causing dark, water-soaked spots with yellow halos. Highly contagious.",
        "severity": "High",
        "immediate_actions": [
            "Remove infected plant parts immediately",
            "Disinfect tools between cuts (70% alcohol)",
            "Avoid overhead watering",
            "Isolate infected plants if possible"
        ],
        "treatment_options": {
            "chemical": [
                "Copper-based bactericides",
                "Streptomycin sulfate (where legal)",
                "Quaternary ammonium compounds"
            ],
            "cultural": [
                "Improve air circulation",
                "Use drip irrigation",
                "Practice crop rotation"
            ]
        },
        "organic_solutions": [
            "Copper fungicide (organic approved)",
            "Hydrogen peroxide solution",
            "Beneficial bacteria applications"
        ],
        "timeline": "Aggressive treatment needed - 14-28 days"
    },

    "downy_mildew": {
        "description": "Oomycete pathogen causing yellow patches with fuzzy gray growth underneath leaves.",
        "severity": "High",
        "immediate_actions": [
            "Remove affected foliage immediately",
            "Improve air circulation dramatically",
            "Reduce humidity levels",
            "Apply systemic fungicide"
        ],
        "treatment_options": {
            "chemical": [
                "Metalaxyl-based systemic fungicides",
                "Fosetyl-aluminum treatments",
                "Dimethomorph applications"
            ],
            "organic": [
                "Copper-based fungicides",
                "Potassium phosphonate"
            ]
        },
        "organic_solutions": [
            "Baking soda with horticultural oil",
            "Milk spray solution",
            "Compost tea applications"
        ],
        "timeline": "Quick action essential - 7-14 days for control"
    },

    "black_spot": {
        "description": "Fungal disease common on roses, causing circular black spots with yellow halos.",
        "severity": "Moderate",
        "immediate_actions": [
            "Remove infected leaves and fallen debris",
            "Prune for better air circulation",
            "Water at soil level only",
            "Apply preventive fungicide"
        ],
        "treatment_options": {
            "chemical": [
                "Myclobutanil fungicide",
                "Propiconazole treatments",
                "Tebuconazole applications"
            ]
        },
        "organic_solutions": [
            "Neem oil spray program",
            "Baking soda solution",
            "Sulfur-based fungicides"
        ],
        "timeline": "10-21 days with consistent treatment"
    },

    "anthracnose": {
        "description": "Fungal disease causing sunken, irregular brown spots, often with pink spore masses.",
        "severity": "Moderate to High",
        "immediate_actions": [
            "Prune out infected branches",
            "Remove fallen leaves and debris",
            "Improve air circulation",
            "Apply fungicide treatment"
        ],
        "treatment_options": {
            "chemical": [
                "Chlorothalonil fungicide",
                "Copper-based treatments",
                "Propiconazole applications"
            ]
        },
        "organic_solutions": [
            "Copper soap fungicide",
            "Neem oil applications",
            "Compost tea with beneficial microbes"
        ],
        "timeline": "14-28 days for control, sanitation crucial"
    },

    "aphid_damage": {
        "description": "Soft-bodied insects that suck plant sap, causing curling, yellowing, and stunted growth.",
        "severity": "Low to Moderate",
        "immediate_actions": [
            "Spray off with strong water jet",
            "Check for ant trails (ants farm aphids)",
            "Apply insecticidal soap",
            "Introduce beneficial insects"
        ],
        "treatment_options": {
            "organic": [
                "Insecticidal soap spray",
                "Neem oil application",
                "Release ladybugs or lacewings"
            ],
            "chemical": [
                "Imidacloprid systemic insecticide",
                "Acetamiprid applications",
                "Thiamethoxam treatments"
            ]
        },
        "organic_solutions": [
            "DIY soap spray (1 tsp dish soap per liter)",
            "Neem oil with pyrethrin",
            "Companion planting with marigolds"
        ],
        "timeline": "3-7 days for population control"
    },

    "spider_mite_damage": {
        "description": "Microscopic pests causing stippling, bronzing, and fine webbing on leaves.",
        "severity": "Moderate",
        "immediate_actions": [
            "Increase humidity around plants",
            "Spray with strong water to dislodge mites",
            "Apply miticide or predatory mites",
            "Improve air circulation"
        ],
        "treatment_options": {
            "chemical": [
                "Abamectin miticide",
                "Bifenthrin treatments",
                "Spiromesifen applications"
            ],
            "biological": [
                "Predatory mite release",
                "Minute pirate bugs",
                "Thrips (beneficial species)"
            ]
        },
        "organic_solutions": [
            "Neem oil spray (weekly)",
            "Rosemary oil solution",
            "Diatomaceous earth dusting"
        ],
        "timeline": "10-14 days with consistent treatment"
    },

    "thrip_damage": {
        "common_name": "Thrip Damage",
        "description": "Tiny insects that rasp leaf surfaces, causing silver streaks and black specks.",
        "severity": "Moderate",
        "immediate_actions": [
            "Use yellow sticky traps",
            "Apply systemic insecticide",
            "Remove heavily damaged leaves",
            "Increase humidity levels"
        ],
        "treatment_options": {
            "chemical": [
                "Spinosad insecticide",
                "Imidacloprid systemic",
                "Acetamiprid applications"
            ]
        },
        "organic_solutions": [
            "Predatory mites release",
            "Neem oil with soap",
            "Reflective mulch application"
        ],
        "timeline": "7-14 days for control"
    },

    "nutrient_deficiency": {
        "description": "Inadequate nutrition causing various symptoms like chlorosis, purpling, or stunted growth.",
        "severity": "Low to Moderate",
        "immediate_actions": [
            "Conduct soil test",
            "Apply appropriate fertilizer",
            "Check and adjust soil pH",
            "Ensure proper drainage"
        ],
        "treatment_options": {
            "nutritional": [
                "Balanced NPK fertilizer",
                "Specific micronutrient supplements",
                "Foliar feeding for quick response"
            ]
        },
        "organic_solutions": [
            "Compost application",
            "Seaweed extract spray",
            "Bone meal for phosphorus",
            "Epsom salt for magnesium"
        ],
        "timeline": "7-21 days for visible improvement"
    }
}

# --- Enhanced Image Processing ---

def load_image(img):
    """Load image from various input types."""
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    elif hasattr(img, "convert"):
        return img.convert("RGB")
    else:
        return Image.open(io.BytesIO(img)).convert("RGB")

def advanced_leaf_segmentation(pil_img):
    """Advanced leaf segmentation with multiple approaches."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        original_shape = bgr.shape[:2]

        # Method 1: Enhanced HSV color segmentation
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Multiple green ranges for different leaf conditions
        green_ranges = [
            ([25, 40, 40], [85, 255, 255]),    # Healthy green
            ([15, 25, 25], [100, 255, 255]),   # Yellow-green (stressed)
            ([5, 30, 30], [35, 255, 255]),     # Yellow (diseased)
            ([85, 40, 40], [100, 255, 255])    # Blue-green
        ]

        combined_mask = np.zeros(original_shape, dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            # Ensure mask is single channel
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Method 2: Brown/diseased tissue detection
        brown_lower = np.array([8, 50, 20], dtype=np.uint8)
        brown_upper = np.array([25, 255, 200], dtype=np.uint8)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        # Ensure brown_mask is single channel
        if len(brown_mask.shape) > 2:
            brown_mask = cv2.cvtColor(brown_mask, cv2.COLOR_BGR2GRAY)
        combined_mask = cv2.bitwise_or(combined_mask, brown_mask)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find the largest contour (main leaf)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest_contour) / (original_shape[0] * original_shape[1])

            # Only proceed if we found a reasonably sized leaf
            if area_ratio > 0.05:  # At least 5% of image
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Add smart padding
                pad_x = int(0.1 * w)
                pad_y = int(0.1 * h)

                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(bgr.shape[1], x + w + pad_x)
                y1 = min(bgr.shape[0], y + h + pad_y)

                cropped = pil_img.crop((x0, y0, x1, y1))
                return cropped, True

        # If segmentation failed, return original
        return pil_img, False

    except Exception as e:
        # If any error occurs, return original image
        return pil_img, False

def enhanced_predict(pil_img):
    """Enhanced prediction with multi-stage analysis."""

    # Stage 1: Preprocess with multiple augmentations for robustness
    predictions = []

    # Original image
    inputs = processor(text=all_prompts, images=pil_img, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions.append(probs)

    # Stage 2: Color-balanced version
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    balanced_bgr = gray_world_cc(bgr)
    balanced_img = Image.fromarray(cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2RGB))

    inputs_balanced = processor(text=all_prompts, images=balanced_img, return_tensors="pt", padding=True)
    inputs_balanced = {k: v.to(device) for k, v in inputs_balanced.items()}

    with torch.no_grad():
        outputs_balanced = model(**inputs_balanced)
        logits_balanced = outputs_balanced.logits_per_image[0]
        probs_balanced = torch.softmax(logits_balanced, dim=-1).cpu().numpy()
        predictions.append(probs_balanced)

    # Average predictions for robustness
    avg_probs = np.mean(predictions, axis=0)

    # Group probabilities by disease label
    label_probs = {}
    for i, prompt in enumerate(all_prompts):
        label = prompt_to_label[prompt]
        if label not in label_probs:
            label_probs[label] = []
        label_probs[label].append(avg_probs[i])

    # Take maximum probability within each group
    final_probs = {}
    for label, probs_list in label_probs.items():
        final_probs[label] = max(probs_list)

    # Normalize probabilities
    total_prob = sum(final_probs.values())
    if total_prob > 0:
        for label in final_probs:
            final_probs[label] /= total_prob

    return final_probs

# --- Main Analysis Function ---

def analyze_plant_disease(image_input):
    """Comprehensive plant disease analysis with validation."""

    try:
        if image_input is None:
            return None, "❌ **Please upload an image to analyze.**"

        # Load and validate image
        pil_img = load_image(image_input)

        # Check basic image quality
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        brightness, contrast = exposure_score(bgr)
        focus_score = variance_of_laplacian(bgr)

        # Image quality checks
        if brightness < 30:
            return None, "⚠️ **Image is too dark.** Please capture in better lighting conditions."

        if brightness > 240:
            return None, "⚠️ **Image is overexposed.** Please reduce lighting or avoid flash."

        if focus_score < 10:
            return None, "⚠️ **Image is too blurry.** Please hold camera steady and ensure proper focus."

        # Plant content validation
        is_plant, green_ratio, edge_density = detect_plant_content(pil_img)

        if not is_plant:
            return None, f"""❌ **No plant or leaf detected in the image.**

**Please ensure your image contains:**
- 🌿 A clear view of plant leaves
- 🌱 Sufficient plant material (not just stems or flowers)
- 📸 Good lighting showing leaf details
- 🎯 Focus on the affected area if disease is suspected

**Current image analysis:**
- Green content: {green_ratio:.1%}
- Edge detail: {'Sufficient' if edge_density > 0.02 else 'Insufficient'}

**Tips for better results:**
- Fill the frame with leaves
- Capture both healthy and affected areas
- Use natural lighting when possible
- Avoid heavy shadows or glare"""

        # Advanced leaf segmentation
        cropped_img, segmentation_success = advanced_leaf_segmentation(pil_img)

        # Apply additional image enhancement
        bgr = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        enhanced_bgr = gray_world_cc(bgr)

        # Slight sharpening for better feature detection (safer approach)
        try:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpening_kernel = kernel * 0.1 + np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
            enhanced_bgr = cv2.filter2D(enhanced_bgr, -1, sharpening_kernel)
            enhanced_bgr = np.clip(enhanced_bgr, 0, 255).astype(np.uint8)
        except Exception:
            # If sharpening fails, use original enhanced image
            pass

        processed_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))

        # Enhanced disease prediction
        disease_probs = enhanced_predict(processed_img)

        # Get top predictions
        sorted_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
        top_disease = sorted_diseases[0]
        top_label, top_confidence = top_disease

        # Confidence thresholds for reliability
        if top_confidence < 0.25:
            return processed_img, f"""⚠️ **Low confidence in diagnosis** ({top_confidence*100:.1f}%)

**Possible reasons:**
- Image quality may not be sufficient
- Disease symptoms are unclear or early stage
- Multiple conditions present
- Uncommon disease not in training data

**Recommendations:**
- Try capturing a clearer, closer image
- Include both affected and healthy leaf areas
- Ensure good lighting without shadows
- Consider consulting a local plant pathologist

**Most likely conditions detected:**
{chr(10).join([f'- {disease_database[label]["common_name"]}: {prob*100:.1f}%' for label, prob in sorted_diseases[:3]])}"""

        # Get disease information
        disease_info = disease_database[top_label]
        treatment_info = TREATMENTS[top_label]

        # Generate comprehensive analysis report
        confidence_level = "High" if top_confidence > 0.7 else "Medium" if top_confidence > 0.5 else "Low"

        result = f"""# 🌿 **Plant Disease Analysis Report**

## 📋 **Primary Diagnosis**
**Disease:** {disease_info['common_name']}
**Scientific Name:** {disease_info['scientific_name']}
**Confidence:** {top_confidence*100:.1f}% ({confidence_level})
**Severity Level:** {treatment_info['severity']}

## 🔍 **Disease Description**
{treatment_info['description']}

## 🚨 **Key Symptoms to Look For:**
{chr(10).join([f'• {symptom}' for symptom in disease_info['symptoms']])}

## ⚡ **Immediate Action Required:**
{chr(10).join([f'• {action}' for action in treatment_info['immediate_actions']])}

## 🌱 **Treatment Options**

### 🥬 **Organic/Natural Solutions:**
{chr(10).join([f'• {solution}' for solution in treatment_info['organic_solutions']])}"""

        # Add specific treatment options if available
        if 'treatment_options' in treatment_info:
            for category, treatments in treatment_info['treatment_options'].items():
                result += f"""

### 🧪 **{category.title()} Treatments:**
{chr(10).join([f'• {treatment}' for treatment in treatments])}"""

        result += f"""

## ⏱️ **Expected Timeline:**
{treatment_info['timeline']}

## 📊 **Alternative Possibilities:**"""

        # Show top 3 alternative diagnoses
        for i, (label, prob) in enumerate(sorted_diseases[1:4], 1):
            alt_disease = disease_database[label]
            result += f"""
**{i}. {alt_disease['common_name']}** ({prob*100:.1f}% confidence)"""

        result += f"""

## 🛡️ **Prevention for Future:**
• Maintain good air circulation around plants
• Water at soil level to avoid wetting foliage
• Practice proper plant spacing
• Remove plant debris regularly
• Monitor plants weekly for early signs
• Use disease-resistant varieties when possible

## ⚠️ **Important Notes:**
- This analysis is for educational purposes only
- For valuable plants or severe infestations, consult a professional
- Always test treatments on a small area first
- Follow all product label instructions carefully
- Consider integrated pest management approaches

**Image Quality Assessment:**
- Focus: {'Excellent' if focus_score > 50 else 'Good' if focus_score > 25 else 'Fair'}
- Lighting: {'Optimal' if 80 < brightness < 180 else 'Acceptable' if 50 < brightness < 220 else 'Suboptimal'}
- Plant Content: {green_ratio:.1%} of image
- Segmentation: {'Successful' if segmentation_success else 'Used full image'}"""

        return processed_img, result

    except Exception as e:
        return None, f"""❌ **Error during analysis:** {str(e)}

**Possible solutions:**
- Check image file format (JPG, PNG supported)
- Ensure image is not corrupted
- Try a different image
- Make sure image contains plant material

**If problem persists, try:**
- Reducing image size
- Using better lighting
- Capturing a clearer image"""

# === Advanced Plant Disease Diagnosis with Enhanced Accuracy ===

# [Previous code remains exactly the same until the Gradio Interface section]

# Gradeo interface
def create_interface():
    """Create the Gradio interface for plant disease diagnosis with high contrast theme."""

    with gr.Blocks(
        title="🌿 Plant Pathology AI Assistant",
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="green",
            neutral_hue="stone",
            font=[gr.themes.GoogleFont("Open Sans")]
        ),
        css="""
        body {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-container {
            max-width: 1000px !important;
            font-family: 'Open Sans', sans-serif;
            background-color: #000000;
            color: #ffffff;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #000000, #047857);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #10B981;
        }
        .upload-area {
            border: 2px dashed #10B981;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            background: #111111;
            margin-bottom: 1rem;
            color: white;
        }
        .tips-box {
            background: #111111;
            border-left: 4px solid #10B981;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
            font-size: 0.9rem;
            color: white;
        }
        .diagnosis-card {
            background: #111111;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
            border: 1px solid #10B981;
            color: white;
        }
        .footer {
            font-size: 0.8rem;
            color: #ffffff;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #10B981;
            background: #111111;
            border-radius: 8px;
            padding: 1rem;
        }
        .hide-buttons .gr-button {
            display: none !important;
        }
        .processed-image {
            border-radius: 8px;
            border: 1px solid #10B981;
            background: #111111;
        }
        .gr-button {
            background: #047857 !important;
            color: white !important;
            border: none !important;
        }
        .gr-button:hover {
            background: #059669 !important;
        }
        .gr-radio-item {
            background: #111111 !important;
            color: white !important;
        }
        .gr-input, .gr-textarea, .gr-dropdown {
            background: #111111 !important;
            color: white !important;
            border: 1px solid #10B981 !important;
        }
        .gr-form {
            background: #000000 !important;
        }
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }
        """
    ) as interface:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="margin-bottom: 0.5rem;">🌱 Plant Pathology AI Assistant</h1>
            <p style="margin: 0; opacity: 0.9;">Educational tool for plant disease identification and management</p>
        </div>
        """)

        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML('<div class="upload-area">')
                    image_input = gr.Image(
                        label="Plant Sample Image",
                        type="pil",
                        height=300,
                        elem_classes="hide-buttons"
                    )
                    gr.HTML('</div>')

                analyze_btn = gr.Button(
                    "Analyze Plant Sample",
                    variant="primary",
                    size="lg"
                )

                gr.HTML("""
                <div class="tips-box">
                    <h4 style="margin-top: 0; color: #10B981;">Laboratory Submission Guidelines</h4>
                    <ul style="margin-bottom: 0;">
                        <li>Submit <strong style="color: #ffffff;">clear, well-lit</strong> images of affected leaves</li>
                        <li>Include both <strong style="color: #ffffff;">healthy and symptomatic</strong> tissue</li>
                        <li>Capture <strong style="color: #ffffff;">upper and lower</strong> leaf surfaces</li>
                        <li>Provide <strong style="color: #ffffff;">multiple angles</strong> when possible</li>
                        <li>For small specimens, use <strong style="color: #ffffff;">scale reference</strong></li>
                    </ul>
                </div>
                """)

            with gr.Column(scale=1):
                with gr.Group():
                    processed_image = gr.Image(
                        label="Microscopic Analysis Preview",
                        height=300,
                        elem_classes="processed-image"
                    )

                with gr.Group():
                    analysis_output = gr.Markdown(
                        label="Pathology Report",
                        value="### Plant Health Diagnostic Report\n\nSubmit a sample image to generate analysis...",
                        elem_classes="diagnosis-card"
                    )

        # Event handlers
        analyze_btn.click(
            fn=analyze_plant_disease,
            inputs=[image_input],
            outputs=[processed_image, analysis_output],
            api_name="analyze_disease"
        )

        # Footer information
        gr.HTML("""
        <div class="footer">
            <h4 style="margin-bottom: 0.5rem; color: #10B981;">Supported Pathologies</h4>
            <div style="columns: 2; column-gap: 2rem; font-size: 0.9rem;">
                <p>• Fungal infections (Mildews, Rusts)</p>
                <p>• Bacterial diseases (Leaf Spot, Blight)</p>
                <p>• Environmental stress (Scorch, Chlorosis)</p>
                <p>• Arthropod damage (Mites, Aphids, Thrips)</p>
                <p>• Nutritional deficiencies (NPK, Micronutrients)</p>
                <p>• Healthy tissue assessment</p>
            </div>

            <div style="margin-top: 1.5rem; display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="margin-bottom: 0.5rem;">
                    <strong style="color: #10B981;">Educational Note:</strong>
                    <span style="color: #ffffff;">This tool serves as a learning aid for plant pathology concepts. Clinical decisions should be verified by certified professionals.</span>
                </div>
                <div>
                    <strong style="color: #10B981;">Academic Version:</strong>
                    <span style="color: #ffffff;">CLIP Model v1.2 | Plant PathDB 2024</span>
                </div>
            </div>
        </div>
        """)

    return interface

# --- Launch Application ---

if __name__ == "__main__":
    print("🚀 Starting Plant Pathology AI Assistant...")
    print("📊 Pathology Database: 12 conditions with management protocols")
    print("🤖 AI Model: CLIP with multi-modal analysis")
    print("🎓 Educational Focus: Student-friendly interface")

    # Create and launch interface
    app = create_interface()

    # Launch with queue for better performance
    app.queue(max_size=20).launch(
        debug=False,
        share=False,
        show_error=True,
        inbrowser=True
    )
