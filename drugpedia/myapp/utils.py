# from transformers import T5Tokenizer, TFT5ForConditionalGeneration, pipeline

# # Initialize the tokenizer and model
# model_name = "t5-base"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = TFT5ForConditionalGeneration.from_pretrained(model_name)
# sentiment_analyzer = pipeline("sentiment-analysis")

# def analyze_patient_review(review):
#     # Analyze sentiment of the review
#     sentiment_result = sentiment_analyzer(review)
#     sentiment_label = sentiment_result[0]['label']
    
#     # Define the context with the provided review
#     context = f"""
#     Patient review: {review}
#     """
    
#     question = "Given the context above, what is the reason for the patient's sentiment?"
    
#     # Prepare the input
#     input_text = f"question: {question} context: {context}"
#     inputs = tokenizer.encode(input_text, return_tensors="tf")

#     # Generate the answer
#     outputs = model.generate(inputs, max_length=150)
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # Determine the sentiment and generate a response
#     if sentiment_label == 'NEGATIVE':
#         response = f"The review is negative. The patient expressed dissatisfaction. Explanation: {answer}"
#     else:
#         response = f"The review is positive. The patient has had a favorable experience. Explanation: {answer}"
    
#     return response  # Return the result instead of printing
