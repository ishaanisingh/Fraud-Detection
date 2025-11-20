# Credit Card Fraud Detection System

A web application that uses Machine Learning to predict fraudulent credit card transactions. The system uses a LightGBM model for high-performance classification and a Django REST Framework backend to process data.

## Tech Stack

* **Backend:** Django, Django REST Framework (Python)
* **Machine Learning:** LightGBM, Pandas, NumPy, Joblib
* **Frontend:** HTML, CSS, JavaScript

## Project Structure

* `backend/frauddetection`: Main Django application.
  * Exposes API endpoints via `views.py` and `serializers.py`.
* `frontend/myapp`: Frontend interface (`index.html`) consuming the API.

## How to Run

### 1. Backend (Django)
Navigate to the backend folder and start the server.

```bash
cd backend/frauddetection
python manage.py runserver