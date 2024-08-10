from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(widget=forms.Textarea, label='Enter your review')
