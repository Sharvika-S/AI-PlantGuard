from flask import Flask, render_template, request, url_for, redirect, flash
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model_path = 'model_checkpoint.keras'  # Replace with the correct path to your saved model
model = load_model(model_path)

# Class labels dictionary (update as needed)
class_labels = {
    "Apple___Apple_scab": 0, "Apple___Black_rot": 1,
    "Apple___Cedar_apple_rust": 2, "Apple___healthy": 3,
    "Blueberry___healthy": 4, "Cherry_(including_sour)___Powdery_mildew": 5,
    "Cherry_(including_sour)___healthy": 6, "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 7,
    "Corn_(maize)___Common_rust_": 8, "Corn_(maize)___Northern_Leaf_Blight": 9, "Corn_(maize)___healthy": 10,
    "Grape___Black_rot": 11, "Grape___Esca_(Black_Measles)": 12, "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 13,
    "Grape___healthy": 14, "Orange___Haunglongbing_(Citrus_greening)": 15, "Peach___Bacterial_spot": 16, "Peach___healthy": 17,
    "Pepper,_bell___Bacterial_spot": 18, "Pepper,_bell___healthy": 19, "Potato___Early_blight": 20, "Potato___Late_blight": 21,
    "Potato___healthy": 22, "Raspberry___healthy": 23, "Soybean___healthy": 24, "Squash___Powdery_mildew": 25, "Strawberry___Leaf_scorch": 26,
    "Strawberry___healthy": 27, "Tomato___Bacterial_spot": 28, "Tomato___Early_blight": 29, "Tomato___Late_blight": 30, "Tomato___Leaf_Mold": 31,
    "Tomato___Septoria_leaf_spot": 32, "Tomato___Spider_mites Two-spotted_spider_mite": 33, "Tomato___Target_Spot": 34, "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 35,
    "Tomato___Tomato_mosaic_virus": 36, "Tomato___healthy": 37
}
# Reverse class labels dictionary for prediction mapping
class_labels_reverse = {v: k for k, v in class_labels.items()}

# Disease suggestions dictionary (update as needed)
disease_suggestions = {
    "Apple___Apple_scab": {
    'Cause': 
        " Venturia inaequalis*, which thrives in cool, wet conditions. "
        "The disease spreads through spores that overwinter on fallen leaves and infected fruit."
    ,
    'Suggestion': 
        "1. Remove and destroy fallen leaves and infected fruits to eliminate sources of infection.\n"
        "2. Apply preventive fungicides at the start of the growing season, focusing on critical periods such as bloom and early fruit development.\n"
        "3. Use resistant apple varieties to minimize susceptibility and reduce dependency on chemical treatments.\n"
        "4. Ensure proper spacing between trees to improve air circulation and reduce humidity levels, which can inhibit fungal growth."
    ,
    'Cure': 
        "1. Apply fungicides such as captan, myclobutanil, or mancozeb according to the manufacturer's instructions.\n"
        "2. Prune trees regularly to improve air circulation and light penetration, reducing favorable conditions for the fungus.\n"
        "3. Monitor trees closely for early signs of the disease and act quickly to prevent further spread."
    
},


    "Apple___Black_rot": {
        "Cause": "Caused by the fungus *Botryosphaeria obtusa*, which infects through wounds or natural openings.",
        "Suggestion": (
            "1. Remove and destroy infected plant parts.\n"
            "2. Avoid wounding trees during pruning or other activities.\n"
            "3. Plant resistant apple varieties to reduce susceptibility."
        ),
        "Cure": (
            "1. Apply fungicides like thiophanate-methyl during early infection stages.\n"
            "2. Use copper-based sprays as a preventive measure."
        )
    },
    "Apple___Cedar_apple_rust": {
        "Cause": "Caused by the fungus *Gymnosporangium juniperi-virginianae*, which alternates between apple and cedar hosts.",
        "Suggestion": (
            "1. Remove nearby cedar trees to break the life cycle of the pathogen.\n"
            "2. Apply fungicides at bud break and throughout the growing season.\n"
            "3. Use resistant apple varieties to reduce disease occurrence."
        ),
        "Cure": (
            "1. Apply fungicides like myclobutanil or mancozeb at the first sign of infection.\n"
            "2. Regularly monitor and treat both apple and cedar hosts for signs of the fungus."
        )
    },
    "Apple___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Maintain proper pruning and orchard hygiene.\n"
            "2. Ensure trees are adequately fertilized and irrigated.\n"
            "3. Monitor regularly for early signs of disease or pest infestations."
        ),
        "Cure": "No cure needed. Continue preventive maintenance to ensure health."
    },
    "Blueberry___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Ensure proper soil pH (4.5-5.5) and adequate drainage.\n"
            "2. Prune old branches to encourage new growth and airflow.\n"
            "3. Monitor for pests like aphids and spider mites regularly."
        ),
        "Cure": "No cure needed. Focus on regular maintenance and inspections."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "Cause": "Caused by the fungus *Podosphaera clandestina*, which thrives in warm, dry conditions with high humidity.",
        "Suggestion": (
            "1. Prune and thin out branches to improve air circulation.\n"
            "2. Avoid overhead watering to minimize leaf wetness.\n"
            "3. Use resistant cherry varieties when possible."
        ),
        "Cure": (
            "1. Apply sulfur-based fungicides or potassium bicarbonate sprays to control the infection.\n"
            "2. Treat with fungicides like myclobutanil during the early stages of the disease."
        )
    },
    "Cherry_(including_sour)___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Ensure proper tree spacing and pruning for adequate airflow.\n"
            "2. Monitor soil nutrient levels and irrigate regularly.\n"
            "3. Inspect trees regularly for early signs of disease or pests."
        ),
        "Cure": "No cure needed. Maintain good practices to sustain health."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "Cause": "Caused by the fungus *Cercospora zeae-maydis*, which thrives in warm, humid conditions and spreads via spores.",
        "Suggestion": (
            "1. Rotate crops with non-host species like soybeans to reduce pathogen survival.\n"
            "2. Avoid overhead irrigation to minimize leaf wetness.\n"
            "3. Select hybrids with genetic resistance to gray leaf spot."
        ),
        "Cure": (
            "1. Apply fungicides such as strobilurins or triazoles at the early stages of infection.\n"
            "2. Use integrated pest management (IPM) practices to reduce disease pressure."
        )
    },
    "Corn_(maize)___Common_rust_": {
        "Cause": "Caused by the fungus *Puccinia sorghi*, it spreads through airborne spores in cool, moist conditions.",
        "Suggestion": (
            "1. Use resistant maize hybrids to minimize susceptibility.\n"
            "2. Avoid excessive nitrogen application, as it promotes dense foliage conducive to disease.\n"
            "3. Monitor crops closely during the early growing season for signs of infection."
        ),
        "Cure": (
            "1. Apply fungicides like propiconazole or pyraclostrobin when rust is first observed.\n"
            "2. Remove severely infected plants to prevent further spore spread."
        )
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Cause": "Caused by the fungus *Exserohilum turcicum*, thriving in moderate temperatures and high humidity.",
        "Suggestion": (
            "1. Rotate crops with non-host species to reduce overwintering pathogens.\n"
            "2. Use resistant hybrids to minimize susceptibility.\n"
            "3. Ensure proper plant spacing to improve airflow and reduce moisture."
        ),
        "Cure": (
            "1. Apply fungicides such as mancozeb or propiconazole at the first sign of infection.\n"
            "2. Remove and destroy infected plant debris after harvest."
        )
    },
    "Corn_(maize)___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Maintain optimal soil fertility and irrigation practices.\n"
            "2. Monitor fields regularly for signs of pest or disease.\n"
            "3. Use crop rotation to sustain soil health and reduce potential disease pressure."
        ),
        "Cure": "No cure needed. Continue with preventive agricultural practices."
    },
    "Grape___Black_rot": {
        "Cause": "Caused by the fungus *Guignardia bidwellii*, it spreads through rain-splashed spores.",
        "Suggestion": (
            "1. Prune and remove infected canes, leaves, and mummified berries.\n"
            "2. Avoid planting grapes in areas prone to waterlogging.\n"
            "3. Use grape varieties resistant to black rot."
        ),
        "Cure": (
            "1. Apply fungicides like myclobutanil or mancozeb at early growth stages.\n"
            "2. Ensure adequate sunlight penetration and airflow by training vines properly."
        )
    },
    "Grape___Esca_(Black_Measles)": {
        "Cause": "Caused by a complex of fungi, including *Phaeomoniella chlamydospora* and *Phaeoacremonium spp.*.",
        "Suggestion": (
            "1. Avoid mechanical injuries to vines that facilitate fungal entry.\n"
            "2. Apply protective fungicides to pruning wounds.\n"
            "3. Remove and destroy infected plant parts promptly."
        ),
        "Cure": (
            "1. Treat with systemic fungicides early in the season.\n"
            "2. Manage irrigation to avoid overwatering, which exacerbates symptoms."
        )
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "Cause": "Caused by the fungus *Isariopsis clavispora*, which thrives in warm, humid conditions.",
        "Suggestion": (
            "1. Ensure proper vineyard spacing to improve air circulation.\n"
            "2. Regularly remove fallen leaves and debris to reduce fungal inoculum.\n"
            "3. Use resistant grape varieties where available."
        ),
        "Cure": (
            "1. Apply protective fungicides such as copper-based sprays or mancozeb.\n"
            "2. Monitor for early signs and treat as soon as symptoms appear."
        )
    },
    "Grape___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Regularly prune and maintain vines for optimal health.\n"
            "2. Monitor for signs of disease or pest infestations.\n"
            "3. Ensure balanced fertilization and proper irrigation practices."
        ),
        "Cure": "No cure needed. Continue preventive measures to maintain health."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "Cause": "Caused by a bacterium (*Candidatus Liberibacter*) and spread by the Asian citrus psyllid.",
        "Suggestion": (
            "1. Use certified disease-free planting materials.\n"
            "2. Control the Asian citrus psyllid population using insecticides or natural predators.\n"
            "3. Remove and destroy infected trees to prevent disease spread."
        ),
        "Cure": (
            "1. There is no cure; focus on management through vector control.\n"
            "2. Apply foliar micronutrient sprays to improve tree health and delay symptoms."
        )
    },
    "Peach___Bacterial_spot": {
        "Cause": "Caused by the bacterium *Xanthomonas campestris pv. pruni*, it thrives in warm, wet conditions.",
        "Suggestion": (
            "1. Avoid overhead irrigation to minimize leaf wetness.\n"
            "2. Use resistant peach varieties where available.\n"
            "3. Remove and destroy infected plant material to reduce bacterial spread."
        ),
        "Cure": (
            "1. Apply copper-based bactericides during the growing season.\n"
            "2. Use proper spacing and pruning to enhance air circulation."
        )
    },
    "Peach___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Prune regularly to maintain tree structure and remove weak branches.\n"
            "2. Monitor trees for early signs of pests or disease.\n"
            "3. Apply balanced fertilizers to promote healthy growth."
        ),
        "Cure": "No cure needed. Focus on maintaining good cultural practices."
    },
    "Pepper,_bell___Bacterial_spot": {
        "Cause": "Caused by the bacterium *Xanthomonas campestris pv. vesicatoria*, spread via infected seeds or water.",
        "Suggestion": (
            "1. Use certified disease-free seeds for planting.\n"
            "2. Avoid overhead irrigation to reduce leaf wetness.\n"
            "3. Rotate crops with non-host species to break the disease cycle."
        ),
        "Cure": (
            "1. Apply copper-based bactericides to manage bacterial infections.\n"
            "2. Remove and destroy infected plants to prevent further spread."
        )
    },
    "Pepper,_bell___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Use disease-resistant varieties and ensure proper spacing for air circulation.\n"
            "2. Monitor plants regularly for early signs of disease or pests.\n"
            "3. Use appropriate fertilization and irrigation practices."
        ),
        "Cure": "No cure needed. Continue preventive agricultural measures."
    },
    "Potato___Early_blight": {
        "Cause": "Caused by the fungus *Alternaria solani*, typically affecting older foliage in warm, wet conditions.",
        "Suggestion": (
            "1. Rotate crops with non-host plants to reduce fungal spores in the soil.\n"
            "2. Use certified disease-free seed potatoes to prevent initial infections.\n"
            "3. Apply mulch to minimize soil splashing onto leaves during irrigation or rainfall."
        ),
        "Cure": (
            "1. Apply fungicides such as chlorothalonil or mancozeb when the first symptoms appear.\n"
            "2. Remove and destroy infected plant debris to reduce fungal inoculum."
        )
    },
    "Potato___Late_blight": {
        "Cause": "Caused by the water mold *Phytophthora infestans*, which thrives in cool, moist conditions.",
        "Suggestion": (
            "1. Plant resistant potato varieties and use certified disease-free seed potatoes.\n"
            "2. Ensure proper spacing between plants to enhance air circulation.\n"
            "3. Monitor weather conditions and apply fungicides preventively if necessary."
        ),
        "Cure": (
            "1. Use fungicides like chlorothalonil or metalaxyl to control the spread.\n"
            "2. Remove and destroy infected plants immediately to prevent disease spread."
        )
    },
    "Potato___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Monitor plants regularly for early signs of disease or pest infestation.\n"
            "2. Use crop rotation and balanced fertilization to maintain soil health.\n"
            "3. Avoid overwatering and ensure proper drainage."
        ),
        "Cure": "No cure needed. Continue with good agricultural practices."
    },
    "Raspberry___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Regularly prune raspberry canes to improve air circulation.\n"
            "2. Apply mulch to retain soil moisture and prevent weeds.\n"
            "3. Monitor for pests or signs of disease to act promptly if needed."
        ),
        "Cure": "No cure needed. Focus on preventive care and maintenance."
    },
    "Soybean___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Rotate crops with non-leguminous plants to prevent disease buildup.\n"
            "2. Maintain proper soil fertility and use disease-resistant varieties.\n"
            "3. Monitor fields for signs of pest or disease and take immediate action if required."
        ),
        "Cure": "No cure needed. Maintain good agricultural practices for sustained health."
    },
    "Squash___Powdery_mildew": {
        "Cause": "Caused by fungi such as *Podosphaera xanthii*, spreading in warm and dry conditions.",
        "Suggestion": (
            "1. Use resistant squash varieties and ensure proper plant spacing for airflow.\n"
            "2. Avoid overhead irrigation to reduce leaf wetness.\n"
            "3. Remove and destroy infected leaves promptly."
        ),
        "Cure": (
            "1. Apply sulfur-based or potassium bicarbonate fungicides to control the disease.\n"
            "2. Maintain regular monitoring to catch infections early."
        )
    },
    "Strawberry___Leaf_scorch": {
        "Cause": "Caused by the fungus *Diplocarpon earliana*, thriving in warm, wet conditions.",
        "Suggestion": (
            "1. Remove and destroy infected leaves to prevent the spread of fungal spores.\n"
            "2. Avoid overhead irrigation and ensure proper plant spacing.\n"
            "3. Use resistant strawberry varieties if available."
        ),
        "Cure": (
            "1. Apply fungicides such as captan or myclobutanil at the first sign of infection.\n"
            "2. Implement crop rotation with non-host plants to reduce disease incidence."
        )
    },
    "Strawberry___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Regularly monitor plants for early signs of disease or pests.\n"
            "2. Apply mulch to retain moisture and prevent weeds.\n"
            "3. Use balanced fertilization to promote healthy growth."
        ),
        "Cure": "No cure needed. Focus on preventive agricultural practices."
    },
    "Tomato___Bacterial_spot": {
        "Cause": "Caused by bacteria *Xanthomonas spp.*, spread through water, tools, or infected seeds.",
        "Suggestion": (
            "1. Use certified disease-free seeds and resistant tomato varieties.\n"
            "2. Avoid overhead irrigation and practice proper crop rotation.\n"
            "3. Sterilize tools and equipment to prevent bacterial spread."
        ),
        "Cure": (
            "1. Apply copper-based bactericides to manage bacterial infections.\n"
            "2. Remove and destroy infected plants to reduce further spread."
        )
    },
    "Tomato___Early_blight": {
        "Cause": "Caused by the fungus *Alternaria solani*, infecting older foliage first in warm, wet conditions.",
        "Suggestion": (
            "1. Remove and destroy infected leaves to minimize spore spread.\n"
            "2. Use disease-free seeds and rotate crops regularly.\n"
            "3. Apply mulch to prevent soil splashing onto leaves."
        ),
        "Cure": (
            "1. Apply fungicides such as mancozeb or chlorothalonil as needed.\n"
            "2. Maintain regular monitoring and treat plants at the first signs of infection."
        )
    },
    "Tomato___Late_blight": {
        "Cause": "Caused by the water mold *Phytophthora infestans*, thriving in cool, moist environments.",
        "Suggestion": (
            "1. Use disease-resistant tomato varieties and certified disease-free seeds.\n"
            "2. Space plants adequately to improve air circulation and reduce humidity.\n"
            "3. Monitor weather conditions and apply preventive fungicides if necessary."
        ),
        "Cure": (
            "1. Apply fungicides such as chlorothalonil or metalaxyl at the first sign of infection.\n"
            "2. Remove and destroy infected plants immediately to halt disease spread."
        )
    },
    "Tomato___Leaf_Mold": {
        "Cause": "Caused by the fungus *Passalora fulva*, which thrives in high humidity and poor ventilation.",
        "Suggestion": (
            "1. Ensure proper ventilation in greenhouses and avoid overcrowding plants.\n"
            "2. Water plants at the base to reduce leaf wetness and humidity.\n"
            "3. Use resistant tomato varieties and remove infected leaves promptly."
        ),
        "Cure": (
            "1. Apply fungicides such as copper-based sprays or chlorothalonil.\n"
            "2. Maintain regular monitoring to detect and treat early signs of infection."
        )
    },
    "Tomato___Septoria_leaf_spot": {
        "Cause": "Caused by the fungus *Septoria lycopersici*, spreading through water splashes or infected plant debris.",
        "Suggestion": (
            "1. Avoid overhead irrigation and provide adequate plant spacing to improve airflow.\n"
            "2. Remove and destroy infected plant debris and foliage.\n"
            "3. Practice crop rotation with non-host plants to prevent reinfection."
        ),
        "Cure": (
            "1. Apply fungicides like mancozeb or chlorothalonil to control the disease.\n"
            "2. Monitor plants regularly and treat infections early to minimize damage."
        )
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "Cause": "Caused by the *Tetranychus urticae* mites, which feed on plant sap, causing yellowing and leaf stippling.",
        "Suggestion": (
            "1. Introduce natural predators like ladybugs or predatory mites to control spider mites.\n"
            "2. Spray water on the underside of leaves to dislodge mites.\n"
            "3. Maintain proper irrigation to reduce plant stress and susceptibility."
        ),
        "Cure": (
            "1. Apply miticides or horticultural oils like neem oil to control mite populations.\n"
            "2. Remove heavily infested leaves to prevent the spread of mites."
        )
    },
    "Tomato___Target_Spot": {
        "Cause": "Caused by the fungus *Corynespora cassiicola*, which spreads through infected plant debris or water splashes.",
        "Suggestion": (
            "1. Avoid overhead irrigation and ensure proper spacing between plants.\n"
            "2. Remove and destroy infected plant debris and leaves.\n"
            "3. Rotate crops with non-host plants to reduce fungal buildup in the soil."
        ),
        "Cure": (
            "1. Apply fungicides such as mancozeb or azoxystrobin to control the disease.\n"
            "2. Monitor plants regularly and treat early to minimize yield loss."
        )
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "Cause": "Caused by the *Begomovirus*, transmitted by whiteflies.",
        "Suggestion": (
            "1. Use resistant tomato varieties and control whitefly populations with yellow sticky traps.\n"
            "2. Apply reflective mulches to repel whiteflies and prevent virus transmission.\n"
            "3. Remove and destroy infected plants to reduce virus spread."
        ),
        "Cure": (
            "1. Use insecticides like imidacloprid to control whiteflies effectively.\n"
            "2. Maintain field hygiene and avoid planting near infected areas."
        )
    },
    "Tomato___Tomato_mosaic_virus": {
        "Cause": "Caused by the *Tobamovirus*, spreading through infected seeds, soil, or tools.",
        "Suggestion": (
            "1. Use certified disease-free seeds and sanitize tools before use.\n"
            "2. Avoid handling plants when wet to prevent virus spread.\n"
            "3. Rotate crops and avoid planting tomatoes near infected areas."
        ),
        "Cure": (
            "1. Remove and destroy infected plants to prevent further spread.\n"
            "2. Focus on prevention as there is no direct chemical cure for the virus."
        )
    },
    "Tomato___healthy": {
        "Cause": "No disease detected.",
        "Suggestion": (
            "1. Continue monitoring plants regularly for early signs of disease or pest infestation.\n"
            "2. Use balanced fertilization and ensure proper watering practices.\n"
            "3. Maintain good air circulation and avoid overwatering to prevent potential issues."
        ),
        "Cure": "No cure needed. Continue good agricultural practices to ensure plant health."
    }
}

# Helper function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route('/')
def home():
    return render_template('index.html', image_url=None, result=None, suggestion=None, cause=None, cure=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image and make predictions
        image_array = preprocess_image(file_path)
        prediction = model.predict(image_array, verbose=0)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_labels_reverse.get(predicted_class_index, "Unknown")
        confidence_score = np.max(prediction) * 100  # Convert to percentage
        
        suggestion = disease_suggestions.get(predicted_class_name, {})
        cause = suggestion.get('Cause', '').replace("\n", "<br>")
        suggestion_text = suggestion.get('Suggestion', '').replace("\n", "<br>")
        cure = suggestion.get('Cure', '').replace("\n", "<br>")
        
        return render_template(
            'index.html', 
            image_url=url_for('static', filename=f'uploads/{filename}'), 
            result=predicted_class_name, 
            confidence=f"{confidence_score:.2f}%", 
            suggestion=suggestion_text, 
            cause=cause, 
            cure=cure, 
            error=None
        )
    else:
        return render_template('index.html', error="Only image files are supported.")

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        if name and email and message:
            # Here, you can save feedback to a database or send it via email
            flash("Thank you for your feedback!", "success")
            return redirect(url_for('home'))
        else:
            flash("All fields are required!", "error")
            return redirect(url_for('feedback'))
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
