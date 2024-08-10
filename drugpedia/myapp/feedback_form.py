# feedback_form.py
from django import forms

class FeedbackForm(forms.Form):
    drugname = forms.CharField(label='Drug Name', max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))
    condition = forms.CharField(label='Condition', max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))
    review = forms.CharField(label='Review', widget=forms.Textarea(attrs={'class': 'form-control'}))

