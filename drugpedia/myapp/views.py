from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from .load_model import (
    condition_model,
    rating_model,
    sentiment_model,
    drug_exploration_model,
    arima_model,
    tfidf_vectorizer_condition,
    count_vectorizer_condition,
    tfidf_vectorizer_rating,
    count_vectorizer_rating,
    tokenizer,
    tokenizer_explore,
    combined_vectorizer,
    label_encoder_condition,
    label_encoder,
    le_condition,
    le_drug,
    lda_model,
    nmf_model,
    tokenizer
)

import numpy as np
from scipy.sparse import hstack
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


def predict_condition(request):
    predicted_condition = None
    if request.method == "GET":
        review = request.GET.get('review', '')
        if review:
            review_tfidf = tfidf_vectorizer_condition.transform([review])
            review_count = count_vectorizer_condition.transform([review])
            review_combined = hstack([review_tfidf, review_count]).toarray()

            prediction = condition_model.predict(review_combined)
            predicted_condition = label_encoder_condition.inverse_transform(np.argmax(prediction, axis=1))[0]
            print(f"Predicted Condition: {predicted_condition}")  # Debug print

    return render(request, 'predict_condition.html', {'predicted_condition': predicted_condition})

                                                      
data = pd.read_csv(r"C:\Users\devis\mydrugpro\drugpedia\drug_cleanedtrain.csv")

# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
# views.py
from .auth_forms import CustomUserCreationForm

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

def login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout(request):
    auth_logout(request)
    return redirect('login')

@login_required
def home(request):
    return render(request, 'home.html')

from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy

class CustomLoginView(LoginView):
    redirect_authenticated_user = True
    template_name = 'registration/login.html'

    def get_redirect_url(self):
        return reverse_lazy('home')




from django.shortcuts import render, redirect
from .feedback_form import FeedbackForm
from .models import Feedback  # Import the Feedback model

def feedback_form(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            # Process the form data here (e.g., save to database)
            drugname = form.cleaned_data['drugname']
            condition = form.cleaned_data['condition']
            review = form.cleaned_data['review']
            
            # Save the feedback to the database
            Feedback.objects.create(drugname=drugname, condition=condition, review=review)
            
            return redirect('feedback_thanks')  # Redirect to a thank you page after submission
    else:
        form = FeedbackForm()
    return render(request, 'feedback_form.html', {'form': form})

def feedback_thanks(request):
    return render(request, 'feedback_thanks.html')




import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
import matplotlib.pyplot as plt
import io
import base64
import random

# Load datasets
df_kerala = pd.read_csv(r"C:\Users\devis\mydrugpro\drugpedia\landslide.csv", encoding='latin')
df_kerala['Date'] = pd.to_datetime(df_kerala['Date'], format='%d-%m-%Y', errors='coerce')
df_kerala.sort_values(by=['Drug', 'Date'], inplace=True)
df_kerala.ffill(inplace=True)

df_up = pd.read_csv(r"C:\Users\devis\mydrugpro\drugpedia\Earthquake.csv", encoding='latin')
df_up['Date'] = pd.to_datetime(df_up['Date'], format='%d-%m-%Y', errors='coerce')
df_up.sort_values(by=['Drug', 'Date'], inplace=True)
df_up.ffill(inplace=True)

def forecast_demand(df, drug_name, periods=3, growth_rate=0.1):
    drug_data = df[df['Drug'] == drug_name]
    drug_data.set_index('Date', inplace=True)
    drug_data = drug_data['Sales']
    
    # Ensure the index frequency is set
    drug_data.index = pd.DatetimeIndex(drug_data.index).to_period('M')
    
    # Fit ARIMA model
    if len(drug_data) < 6:  # Check for minimum number of data points
        raise ValueError(f"Not enough data points for {drug_name}. At least 6 data points are required.")
    
    # Use preloaded model
    model_fit = arima_model  # Ensure this is defined somewhere in your script
    
    # Forecast
    forecast = model_fit.forecast(steps=periods)
    
    # Apply exponential growth factor
    for i in range(1, periods + 1):
        forecast.iloc[i-1] *= (1 + growth_rate) ** i
    
    # Round forecast to the nearest integer
    forecast = forecast.round().astype(int)
    
    return drug_data, forecast

def get_random_reviews(df, drug_name, num_reviews=5):
    reviews = df[df['Drug'] == drug_name].sample(n=num_reviews)
    return reviews[['Review', 'Price', 'Sentiment']].to_dict(orient='records')

def forecast_form(request):
    return render(request, 'forecast_form.html')

def forecast(request):
    if request.method == 'POST':
        drug_name = request.POST.get('drug_name')
        periods = int(request.POST.get('periods', 3))
        growth_rate = float(request.POST.get('growth_rate', 0.1))
        
        # Verify column names
        print(df_kerala.columns)  # Debug print
        
        # Forecast demand
        try:
            actual_sales, forecast = forecast_demand(df_kerala, drug_name, periods, growth_rate)
        except KeyError as e:
            return HttpResponse(f"Column not found: {e}", status=400)
        
        # Get random reviews
        reviews = get_random_reviews(df_kerala, drug_name)
        
        # Generate plot
        plt.figure(figsize=(10, 5))
        plt.plot(actual_sales.index.to_timestamp(), actual_sales, label='Actual Sales')
        plt.plot(forecast.index.to_timestamp(), forecast, label='Forecast', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Sales Forecast for {drug_name}')
        plt.legend()
        plt.grid(True)

        # Save plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode image as base64
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Prepare forecast data for rendering
        forecast_data = forecast.reset_index()
        forecast_data.columns = ['Date', 'Forecast']

        return render(request, 'forecast_result.html', {
            'drug_name': drug_name,
            'forecast': forecast_data.to_dict(orient='records'),
            'plot_url': plot_url,
            'reviews': reviews
        })
    else:
        return redirect('forecast_form')

def forecast1_form(request):
    return render(request, 'forecast1_form.html')

def forecast1(request):
    if request.method == 'POST':
        drug_name = request.POST.get('drug_name')
        periods = int(request.POST.get('periods', 3))
        growth_rate = float(request.POST.get('growth_rate', 0.1))
        
        # Verify column names
        print(df_up.columns)  # Debug print
        
        # Forecast demand
        try:
            actual_sales, forecast = forecast_demand(df_up, drug_name, periods, growth_rate)
        except KeyError as e:
            return HttpResponse(f"Column not found: {e}", status=400)
        
        # Get random reviews
        reviews = get_random_reviews(df_up, drug_name)
        
        # Generate plot
        plt.figure(figsize=(10, 5))
        plt.plot(actual_sales.index.to_timestamp(), actual_sales, label='Actual Sales')
        plt.plot(forecast.index.to_timestamp(), forecast, label='Forecast', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Sales Forecast for {drug_name}')
        plt.legend()
        plt.grid(True)

        # Save plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode image as base64
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Prepare forecast data for rendering
        forecast_data = forecast.reset_index()
        forecast_data.columns = ['Date', 'Forecast']

        return render(request, 'forecast1_result.html', {
            'drug_name': drug_name,
            'forecast': forecast_data.to_dict(orient='records'),
            'plot_url': plot_url,
            'reviews': reviews
        })
    else:
        return redirect('forecast1_form')

def region_forecast(request, region_name):
    if region_name.lower() == "kerala":
        return redirect('forecast_form')
    elif region_name.lower() == "uttar pradesh":
        return redirect('forecast1_form')
    else:
        return HttpResponse("Region not found", status=404)

def home(request):
    return render(request, 'home.html')



import pandas as pd
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Frame, PageTemplate
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from PIL import Image

# Load the medicine dataset with dtype specification
medicine_dataset_path = r"C:\Users\devis\mydrugpro\drugpedia\medicine_dataset.csv"  # Update this path as needed
dtype_spec = {42: str, 43: str, 44: str, 45: str, 46: str, 47: str, 48: str}  # Specify appropriate columns
medicine_data = pd.read_csv(medicine_dataset_path, dtype=dtype_spec)

# Paths to images
background_image_first_page_path = r"C:\Users\devis\mydrugpro\drugpedia\reportfront.jpg"  # Background image 1
background_image_other_pages_path = r"C:\Users\devis\mydrugpro\drugpedia\reportback.jpg"  # Background image 2

# Check if the image files are valid
try:
    for image_path in [background_image_first_page_path, background_image_other_pages_path]:
        with open(image_path, 'rb') as img_file:
            img = Image.open(img_file)
            img.verify()  # Verify that it is an image
except (IOError, FileNotFoundError, Image.UnidentifiedImageError) as e:
    print(f"Error with the image file: {e}")
    exit()

def generate_pdf_report(drug_name):
    drug_info = medicine_data[medicine_data['name'].str.contains(drug_name, case=False, na=False)]
    if drug_info.empty:
        return None

    file_name = f"{drug_name}_report.pdf"
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'

    doc = SimpleDocTemplate(response, pagesize=letter)

    # Styles
    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(name='Normal', fontName='Helvetica', fontSize=12, spaceAfter=12)
    header_style = ParagraphStyle(name='Header', fontName='Helvetica-Bold', fontSize=14, spaceAfter=12)

    # Story for the document content
    story = []

    # Add initial spacer to ensure the text starts from the white area
    story.append(Spacer(1, 3.5 * inch))

    # Drug Name and ID
    drug_name_heading = Paragraph(f"<b>Drug Name:</b> {drug_info['name'].values[0]}", header_style)
    drug_id = Paragraph(f"<b>ID:</b> {drug_info['id'].values[0]}", normal_style)
    story.append(drug_name_heading)
    story.append(drug_id)
    story.append(Spacer(1, 12))

    # Use
    use_heading = Paragraph("<b>Use:</b>", header_style)
    use_content = Paragraph(" ".join(drug_info[[col for col in drug_info.columns if 'use' in col]].dropna(axis=1).values[0]), normal_style)
    story.append(use_heading)
    story.append(use_content)
    story.append(Spacer(1, 12))

    # Substitutes
    substitutes_heading = Paragraph("<b>Substitutes:</b>", header_style)
    substitutes_list = drug_info[[col for col in drug_info.columns if 'substitute' in col]].dropna(axis=1).values[0]
    story.append(substitutes_heading)
    for substitute in substitutes_list:
        story.append(Paragraph(substitute, normal_style))
    story.append(Spacer(1, 12))

    # Side Effects
    side_effects_heading = Paragraph("<b>Side Effects:</b>", header_style)
    side_effects_list = drug_info[[col for col in drug_info.columns if 'sideEffect' in col]].dropna(axis=1).values[0]
    story.append(side_effects_heading)
    for side_effect in side_effects_list:
        story.append(Paragraph(side_effect, normal_style))
    story.append(Spacer(1, 12))

    # Chemical Class
    chemical_class_heading = Paragraph("<b>Chemical Class:</b>", header_style)
    chemical_class_value = drug_info['Chemical Class'].values[0] if pd.notna(drug_info['Chemical Class'].values[0]) else "Not Available"
    chemical_class_content = Paragraph(chemical_class_value, normal_style)
    story.append(chemical_class_heading)
    story.append(chemical_class_content)
    story.append(Spacer(1, 12))

    # Habit Forming
    habit_forming_heading = Paragraph("<b>Habit Forming:</b>", header_style)
    habit_forming_content = Paragraph(str(drug_info['Habit Forming'].values[0]), normal_style)
    story.append(habit_forming_heading)
    story.append(habit_forming_content)
    story.append(Spacer(1, 12))

    # Therapeutic Class
    therapeutic_class_heading = Paragraph("<b>Therapeutic Class:</b>", header_style)
    therapeutic_class_content = Paragraph(str(drug_info['Therapeutic Class'].values[0]), normal_style)
    story.append(therapeutic_class_heading)
    story.append(therapeutic_class_content)
    story.append(Spacer(1, 12))

    # Action Class
    action_class_heading = Paragraph("<b>Action Class:</b>", header_style)
    action_class_value = drug_info['Action Class'].values[0] if pd.notna(drug_info['Action Class'].values[0]) else "Not Available"
    action_class_content = Paragraph(action_class_value, normal_style)
    story.append(action_class_heading)
    story.append(action_class_content)
    story.append(Spacer(1, 12))

    # Function to add the first page background
    def add_first_page_background(canvas, doc):
        canvas.drawImage(background_image_first_page_path, 0, 0, width=letter[0], height=letter[1])

    # Function to add the other pages background
    def add_other_pages_background(canvas, doc):
        canvas.drawImage(background_image_other_pages_path, 0, 0, width=letter[0], height=letter[1])

    # Define the white space frame and move the text slightly up
    frame = Frame(inch, 3 * inch, letter[0] - 2 * inch, letter[1] - 3.5 * inch, id='white_space_frame')

    # Define the page templates with background and frame
    first_page_template = PageTemplate(id='FirstPage', frames=[frame], onPage=add_first_page_background)
    other_pages_template = PageTemplate(id='OtherPages', frames=[frame], onPage=add_other_pages_background)
    doc.addPageTemplates([first_page_template, other_pages_template])

    # Build the PDF
    doc.build(story)

    return response

def download_report(request):
    drug_name = request.GET.get('drug_name', '')
    if drug_name:
        response = generate_pdf_report(drug_name)
        if response:
            return response
        else:
            return HttpResponse("No information found for the drug.", status=404)
    return HttpResponse("Drug name not provided.", status=400)


import tensorflow as tf
import numpy as np
import joblib
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from io import BytesIO
import base64
import pandas as pd
from django.shortcuts import render

# Download stopwords from nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def get_stop_words():
    return set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

def preprocess(text, stop_words):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def compute_integrated_gradients(model, review_input, baseline_input, num_steps=50):
    baseline_input = tf.convert_to_tensor(baseline_input, dtype=tf.float32)
    review_input = tf.convert_to_tensor(review_input, dtype=tf.float32)
    grad_list = []

    for i in range(num_steps + 1):
        alpha = tf.constant(i / num_steps, dtype=tf.float32)
        interpolated_input = baseline_input + alpha * (review_input - baseline_input)
        with tf.GradientTape() as tape:
            tape.watch(interpolated_input)
            prediction = model(interpolated_input, training=False)
        grads = tape.gradient(prediction, interpolated_input)
        grad_list.append(grads.numpy())

    avg_grads = np.mean(np.array(grad_list), axis=0)
    integrated_gradients = (review_input - baseline_input) * avg_grads
    return integrated_gradients

def visualize_review(review, integrated_gradients, vectorizer):
    words = review.split()
    tfidf_features = vectorizer.get_feature_names_out()
    word_importances = {}

    for i, word in enumerate(words):
        if word in tfidf_features:
            idx = vectorizer.vocabulary_.get(word)
            if idx is not None:
                word_importances[word] = integrated_gradients[0][idx]

    max_importance = max(np.abs(list(word_importances.values())), default=1)
    word_importances = {word: importance / max_importance for word, importance in word_importances.items()}

    html_output = ""
    colors = ["yellow", "orange", "blue", "lightgreen", "pink"]
    color_index = 0
    for word in words:
        if word in word_importances:
            importance = word_importances[word]
            color = colors[color_index % len(colors)]
            color_index += 1
            html_output += f'<span style="background-color:{color}; color:white;">{word}</span> '
        else:
            html_output += f'<span style="color:white;">{word}</span> '

    return html_output, word_importances

def predict_rating(request):
    predicted_rating = None
    html_output = None
    image_base64 = None

    if request.method == "GET":
        review = request.GET.get('review', '')
        if review:
            stop_words = get_stop_words()
            review_cleaned = preprocess(review, stop_words)

            review_tfidf = tfidf_vectorizer_rating.transform([review_cleaned]).toarray()
            review_count = count_vectorizer_rating.transform([review_cleaned]).toarray()
            review_combined = np.hstack([review_tfidf, review_count])

            baseline_input = np.zeros_like(review_combined)

            integrated_gradients = compute_integrated_gradients(rating_model, review_combined, baseline_input)

            html_output, word_importances = visualize_review(review_cleaned, integrated_gradients, tfidf_vectorizer_rating)

            sorted_importances = sorted(word_importances.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
            top_features = [item[0] for item in sorted_importances]
            top_scores = [abs(item[1]).numpy() for item in sorted_importances]

            df = pd.DataFrame({
                'Feature': top_features,
                'Importance': top_scores
            })

            plt.figure(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
            plt.title('Top Features Contributing to the Rating')
            plt.xlabel('Importance')
            plt.ylabel('Feature')

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            image_base64 = base64.b64encode(image_png).decode('utf-8')

            prediction = rating_model.predict(review_combined)
            predicted_rating = prediction[0][0]

    return render(request, 'predict_rating.html', {
        'predicted_rating': predicted_rating,
        'html_output': html_output,
        'image_base64': image_base64,
    })



from django.shortcuts import render
from transformers import T5Tokenizer, TFT5ForConditionalGeneration, pipeline

# Initialize the tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_patient_review(review):
    # Analyze sentiment of the review
    sentiment_result = sentiment_analyzer(review)
    sentiment_label = sentiment_result[0]['label']

    # Define the context with the provided review
    context = f"""
    Patient review: {review}
    """

    question = "Given the context above, what is the reason for the patient's sentiment?"

    # Prepare the input
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="tf")

    # Generate the answer
    outputs = model.generate(inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Determine the sentiment and generate a response
    if sentiment_label == 'NEGATIVE':
        response = f"The review is negative. The patient expressed dissatisfaction. Explanation: {answer}"
    else:
        response = f"The review is positive. The patient has had a favorable experience. Explanation: {answer}"
    return sentiment_label, response

def predict_sentiment(request):
    sentiment_label = None
    explanation = None
    
    if request.method == "GET":
        review = request.GET.get('review', '')
        if review:
            sentiment_label, explanation = analyze_patient_review(review)
            print(f"Predicted Sentiment: {sentiment_label}")  # Debug print
        else:
            sentiment_label = "No review provided"
            explanation = "Please provide a review to analyze."

    return render(request, 'predict_sentiment.html', {'sentiment': sentiment_label, 'explanation': explanation})



from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from rake_nltk import Rake

# Load the data
data = pd.read_csv(r"C:\Users\devis\mydrugpro\drugpedia\drug_cleanedtrain.csv")

def get_reviews_and_conditions(drug_name):
    filtered_data = data[data['drugName'].str.lower() == drug_name.lower()]
    if filtered_data.empty:
        return [], []
    unique_conditions = filtered_data['condition'].unique()[:5]
    reviews = filtered_data['review'].values[:5]
    return unique_conditions, reviews

# Highlight important words in reviews
def highlight_words(text, keywords):
    for word in keywords:
        pattern = rf'\b{re.escape(word)}\b'
        text = re.sub(pattern, f'<span class="highlight" style="background-color: yellow;">{word}</span>', text, flags=re.IGNORECASE)
    return text

# Get top keywords for each topic
def get_top_keywords(model, feature_names, n_top_words):
    keywords = []
    for topic_idx, topic in enumerate(model.components_):
        keywords.extend([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return keywords

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub('\d+', '', text)
    return text

def get_rake_keywords(reviews):
    rake = Rake()
    keywords = []
    for review in reviews:
        rake.extract_keywords_from_text(review)
        keywords.extend(rake.get_ranked_phrases())
    return keywords

def explore_drug(request):
    drug_info = {
        'drugName': None,
        'conditions': [],
        'reviews': []
    }
    if request.method == "GET":
        drug_name = request.GET.get('drug_name', '')
        if drug_name:
            conditions, reviews = get_reviews_and_conditions(drug_name)

            # Preprocess reviews for topic modeling
            preprocessed_reviews = [preprocess_text(review) for review in reviews]
            tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
            count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)
            count_data = count_vectorizer.fit_transform(preprocessed_reviews)

            # NMF and LDA models
            nmf = NMF(n_components=1, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha_W=.1, l1_ratio=.5)
            lda = LDA(n_components=1, random_state=0)
            nmf_topics = nmf.fit_transform(tfidf)
            lda_topics = lda.fit_transform(count_data)

            # Get top keywords
            nmf_keywords = get_top_keywords(nmf, tfidf_vectorizer.get_feature_names_out(), 10)
            lda_keywords = get_top_keywords(lda, count_vectorizer.get_feature_names_out(), 10)
            rake_keywords = get_rake_keywords(preprocessed_reviews)
            ensemble_keywords = list(set(nmf_keywords).union(set(lda_keywords)).union(set(rake_keywords)))

            # Highlight keywords in reviews
            highlighted_reviews = [highlight_words(review, ensemble_keywords) for review in reviews]

            drug_info['drugName'] = drug_name
            drug_info['conditions'] = conditions.tolist() if isinstance(conditions, np.ndarray) else conditions
            drug_info['reviews'] = highlighted_reviews

    return render(request, 'explore_drug.html', {'drug_info': drug_info})

def get_reviews_by_sentiment(drug_name, sentiment, df):
    if sentiment == 'positive':
        filtered_reviews = df[(df['drugName'] == drug_name) & (df['rating'] >= 7)]
    elif sentiment == 'negative':
        filtered_reviews = df[(df['drugName'] == drug_name) & (df['rating'] <= 4)]
    elif sentiment == 'neutral':
        filtered_reviews = df[(df['drugName'] == drug_name) & (df['rating'] == 5)]
    else:
        return ["Invalid sentiment. Choose from 'positive', 'negative', or 'neutral'."]
    return filtered_reviews['review'].head(10).tolist()

def get_sentiment_distribution(drug_name, df):
    sentiments = {
        'positive': df[(df['drugName'] == drug_name) & (df['rating'] >= 7)],
        'negative': df[(df['drugName'] == drug_name) & (df['rating'] <= 4)],
        'neutral': df[(df['drugName'] == drug_name) & (df['rating'] == 5)]
    }
    distribution = {sentiment: len(reviews) for sentiment, reviews in sentiments.items()}
    distribution = {k: int(v) if not pd.isna(v) else 0 for k, v in distribution.items()}
    
    # Create a pie chart
    fig, ax = plt.subplots()
    if any(distribution.values()):
        ax.pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%', startangle=140)
    else:
        ax.pie([1], labels=['No data'], autopct='%1.1f%%', startangle=140)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert plot to PNG image and then to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return distribution, img_base64

def more_drug_info(request):
    if request.method == 'POST':
        drug_name = request.POST.get('drug_name')
        review_type = request.POST.get('review_type')

        df = pd.read_csv(r"C:\Users\devis\mydrugpro\drugpedia\drug_cleanedtrain.csv")
        df = df.dropna(subset=['drugName', 'rating', 'review'])
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])
        
        reviews = get_reviews_by_sentiment(drug_name, review_type, df)

        # Ensure to preprocess and highlight
        preprocessed_reviews = [preprocess_text(review) for review in reviews]
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)
        count_data = count_vectorizer.fit_transform(preprocessed_reviews)

        nmf = NMF(n_components=1, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha_W=.1, l1_ratio=.5)
        lda = LDA(n_components=1, random_state=0)
        nmf_topics = nmf.fit_transform(tfidf)
        lda_topics = lda.fit_transform(count_data)

        nmf_keywords = get_top_keywords(nmf, tfidf_vectorizer.get_feature_names_out(), 10)
        lda_keywords = get_top_keywords(lda, count_vectorizer.get_feature_names_out(), 10)
        rake_keywords = get_rake_keywords(preprocessed_reviews)
        ensemble_keywords = list(set(nmf_keywords).union(set(lda_keywords)).union(set(rake_keywords)))

        # Print for debugging
        print("Keywords:", ensemble_keywords)

        highlighted_reviews = [highlight_words(review, ensemble_keywords) for review in reviews]

        distribution, pie_chart = get_sentiment_distribution(drug_name, df)
        
        return render(request, 'more_drug_info.html', {
            'reviews': highlighted_reviews,
            'drug_name': drug_name,
            'review_type': review_type,
            'distribution': distribution,
            'pie_chart': pie_chart
        })
    
    return render(request, 'more_drug_info.html')




from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
import string

# Load the data
data_path = r"C:\Users\devis\mydrugpro\drugpedia\drug_cleanedtrain.csv"
data = pd.read_csv(data_path)

def get_reviews_and_drugs(condition_name):
    filtered_data = data[data['condition'].str.lower() == condition_name.lower()]
    if filtered_data.empty:
        return [], []
    unique_drugs = filtered_data['drugName'].dropna().unique()[:5]
    reviews = filtered_data['review'].dropna().values[:5]
    return unique_drugs, reviews

def highlight_words_cond(text, keywords):
    for word in keywords:
        text = re.sub(rf'\b{word}\b', f'<span class="highlight" style="background-color: yellow;">{word}</span>', text, flags=re.IGNORECASE)
    return text

def get_top_keywords_cond(model, feature_names, n_top_words):
    keywords = []
    for topic_idx, topic in enumerate(model.components_):
        keywords.extend([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return keywords

def preprocess_text_cond(text):
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub('\d+', '', text)
    return text

def get_top_keywords_for_each_review(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_review = []
    for topic in model.components_:
        keywords_per_review.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return keywords_per_review

def get_reviews_by_sentiment_cond(condition_name, sentiment, df):
    if sentiment == 'positive':
        filtered_reviews = df[(df['condition'] == condition_name) & (df['rating'] >= 7)]
    elif sentiment == 'negative':
        filtered_reviews = df[(df['condition'] == condition_name) & (df['rating'] <= 4)]
    elif sentiment == 'neutral':
        filtered_reviews = df[(df['condition'] == condition_name) & (df['rating'] == 5)]
    else:
        return ["Invalid sentiment. Choose from 'positive', 'negative', or 'neutral'."]
    return filtered_reviews['review'].head(10).tolist()

def get_sentiment_distribution_cond(condition_name, df):
    sentiments = {
        'positive': df[(df['condition'] == condition_name) & (df['rating'] >= 7)],
        'negative': df[(df['condition'] == condition_name) & (df['rating'] <= 4)],
        'neutral': df[(df['condition'] == condition_name) & (df['rating'] == 5)]
    }
    distribution = {sentiment: len(reviews) for sentiment, reviews in sentiments.items()}
    distribution = {k: int(v) if not pd.isna(v) else 0 for k, v in distribution.items()}

    # Create a pie chart
    fig, ax = plt.subplots()
    if any(distribution.values()):
        ax.pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%', startangle=140)
    else:
        ax.pie([1], labels=['No data'], autopct='%1.1f%%', startangle=140)

    ax.axis('equal')

    # Convert plot to PNG image and then to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return distribution, img_base64

def explore_condition(request):
    condition_info = {
        'conditionName': None,
        'drugs': [],
        'reviews': []
    }
    if request.method == "GET":
        condition_name = request.GET.get('condition', '')
        if condition_name:
            drugs, reviews = get_reviews_and_drugs(condition_name)

            # Preprocess reviews for topic modeling
            preprocessed_reviews = [preprocess_text_cond(review) for review in reviews]
            tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
            count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)
            count_data = count_vectorizer.fit_transform(preprocessed_reviews)

            # NMF and LDA models
            nmf = NMF(n_components=1, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha_W=.1, l1_ratio=.5)
            lda = LDA(n_components=1, random_state=0)
            nmf_topics = nmf.fit_transform(tfidf)
            lda_topics = lda.fit_transform(count_data)

            # Get top keywords
            nmf_keywords = get_top_keywords_cond(nmf, tfidf_vectorizer.get_feature_names_out(), 10)
            lda_keywords = get_top_keywords_cond(lda, count_vectorizer.get_feature_names_out(), 10)
            ensemble_keywords = list(set(nmf_keywords).union(set(lda_keywords)))

            # Highlight keywords in reviews
            highlighted_reviews = [highlight_words_cond(review, ensemble_keywords) for review in reviews]

            condition_info['conditionName'] = condition_name
            condition_info['drugs'] = drugs.tolist() if isinstance(drugs, np.ndarray) else drugs
            condition_info['reviews'] = highlighted_reviews

    return render(request, 'explore_condition.html', {'condition_info': condition_info})

def more_condition_info(request):
    if request.method == 'POST':
        condition_name = request.POST.get('condition_name')
        review_type = request.POST.get('review_type')

        df = pd.read_csv(data_path)
        df = df.dropna(subset=['condition', 'rating', 'review'])
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])

        reviews = get_reviews_by_sentiment_cond(condition_name, review_type, df)

        # Ensure to preprocess and highlight
        preprocessed_reviews = [preprocess_text_cond(review) for review in reviews]
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)
        count_data = count_vectorizer.fit_transform(preprocessed_reviews)

        nmf = NMF(n_components=1, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha_W=.1, l1_ratio=.5)
        lda = LDA(n_components=1, random_state=0)
        nmf_topics = nmf.fit_transform(tfidf)
        lda_topics = lda.fit_transform(count_data)

        # Get top keywords for each review
        nmf_keywords_per_review = get_top_keywords_for_each_review(nmf, tfidf_vectorizer, 10)
        lda_keywords_per_review = get_top_keywords_for_each_review(lda, count_vectorizer, 10)

        highlighted_reviews = []
        for i, review in enumerate(reviews):
            combined_keywords = set(nmf_keywords_per_review[i]).union(set(lda_keywords_per_review[i]))
            highlighted_review = highlight_words_cond(review, combined_keywords)
            highlighted_reviews.append(highlighted_review)

        distribution, pie_chart = get_sentiment_distribution_cond(condition_name, df)

        return render(request, 'more_condition_info.html', {
            'reviews': highlighted_reviews,
            'condition_name': condition_name,
            'review_type': review_type,
            'distribution': distribution,
            'pie_chart': pie_chart
        })

    return render(request, 'more_condition_info.html')



from django.shortcuts import render
from .models import Feedback

def recent_reviews(request):
    query = request.GET.get('drug_name', '')
    if query:
        feedback_list = Feedback.objects.filter(drugname__icontains=query)
    else:
        feedback_list = Feedback.objects.all()

    return render(request, 'recent_reviews.html', {'feedback_list': feedback_list})


# Create your views here.
