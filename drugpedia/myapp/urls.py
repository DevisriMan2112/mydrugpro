from django.urls import path

from myapp import admin

from django.urls import path
from .views import predict_condition, predict_rating, predict_sentiment, explore_drug, home, download_report, forecast, forecast_form, region_forecast,more_drug_info, register, home, forecast1_form, region_forecast, forecast1, explore_condition, more_condition_info, feedback_form, feedback_thanks, recent_reviews
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LoginView


urlpatterns = [
    path('', home, name='home'),  # Home page
    path('home/', home, name='home'),  # Optional: Consider removing this if the above is sufficient
    path('login/', LoginView.as_view(redirect_authenticated_user=True, template_name='registration/login.html'), name='login'),
    path('register/', register, name='register'),

    # Feedback paths
    path('feedback/', feedback_form, name='feedback_form'),
    path('feedback/thanks/', feedback_thanks, name='feedback_thanks'),

    # Prediction paths
    path('predict_condition/', predict_condition, name='predict_condition'),
    path('predict_rating/', predict_rating, name='predict_rating'),
    path('predict_sentiment/', predict_sentiment, name='predict_sentiment'),

    # Exploration paths
    path('explore_drug/', explore_drug, name='explore_drug'),
    path('more_drug_info/', more_drug_info, name='more_drug_info'),
    path('explore_condition/', explore_condition, name='explore_condition'),
    path('more_condition_info/', more_condition_info, name='more_condition_info'),

    # Report and Forecast paths
    path('download_report/', download_report, name='download_report'),
    path('forecast_form/', forecast_form, name='forecast_form'),
    path('forecast/', forecast, name='forecast_result'),
    path('forecast1_form/', forecast1_form, name='forecast1_form'),
    path('forecast1/', forecast1, name='forecast1_result'),

    # Region-specific forecasts
    path('region/<str:region_name>/', region_forecast, name='region_forecast'),

    # Recent Reviews
    path('recent-reviews/', recent_reviews, name='recent_reviews'),
]