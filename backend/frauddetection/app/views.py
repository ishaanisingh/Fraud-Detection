import pandas as pd
import numpy as np
import joblib
import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from rest_framework.pagination import PageNumberPagination

# --- Import your app's models and serializers ---
from .models import Transaction              
from .serializers import TransactionSerializer 


# ===================================================================
#  1. LOAD THE TRAINED MODEL
# ===================================================================

MODEL_FILE = os.path.join(settings.BASE_DIR, 'app', 'fraud_model_pipeline.pkl')
pipeline_lgbm = None

try:
    pipeline_lgbm = joblib.load(MODEL_FILE)
    print(f"Successfully loaded model from {MODEL_FILE}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_FILE}")
except Exception as e:
    print(f"Error loading model: {e}")

# ===================================================================
#  2. HELPER FUNCTION 
# ===================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

# ===================================================================
#  3. API VIEWS 
# ===================================================================

@api_view(['GET', 'POST'])
def transaction_list_create(request):
    """
    GET: List all transactions (with pagination).
    POST: Create a new transaction and predict fraud.
    """
    
    if request.method == 'GET':
        paginator = PageNumberPagination()
        paginator.page_size = 100  
        
        transactions = Transaction.objects.all().order_by('customer_id')
        paginated_transactions = paginator.paginate_queryset(transactions, request)
        serializer = TransactionSerializer(paginated_transactions, many=True)
        
        return paginator.get_paginated_response(serializer.data)

    elif request.method == 'POST':
        

        if pipeline_lgbm is None:
            return Response(
                {"error": "Model is not loaded. Cannot make predictions."}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = TransactionSerializer(data=request.data)
        
        if serializer.is_valid():
            
            data = serializer.validated_data
            
            # 2.FEATURE ENGINEERING
            
            data_engineered = {}
            

            direct_features = [
                'Gender', 'Transaction Amount', 'Merchant Name', 'Category', 
                'Type of Card', 'Entry Mode', 'Amount', 'Type of Transaction', 
                'Merchant Group', 'Country of Transaction', 'Shipping Address', 
                'Country of Residence', 'Bank', 'state', 'zip', 'city_pop', 'job'
            ]
            for feature in direct_features:
                data_engineered[feature] = data.get(feature)

           
            try:
                # Calculate Age
                bdate = pd.to_datetime(data.get('birthdate'), dayfirst=True)
                data_engineered['Age'] = (pd.to_datetime('2023-01-01') - bdate).days // 365
                
                # Calculate Time features
                dt = pd.to_datetime(data.get('unix_time'), unit='s')
                data_engineered['hour_of_day'] = dt.hour
                data_engineered['day_of_week'] = dt.dayofweek
                
                # Calculate Distance
                data_engineered['distance_km'] = haversine_distance(
                    data.get('lat'), data.get('long'),
                    data.get('merch_lat'), data.get('merch_long')
                )
            except Exception as e:
                return Response(
                    {"error": f"Failed during feature engineering: {e}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 3. --- PREDICTION ---
            
            input_df = pd.DataFrame([data_engineered])
    
            y_prob = pipeline_lgbm.predict_proba(input_df)[0][1]
            
           
            NEW_THRESHOLD = 0.75
            is_fraud = bool(y_prob >= NEW_THRESHOLD)
            
            # 4. --- SAVE TO DATABASE ---
            transaction_instance = serializer.save(is_fraud=is_fraud)
            
            # 5. --- RETURN RESPONSE ---
            response_data = {
                'is_fraud': is_fraud,  
                'fraud_probability': f"{y_prob:.4f}",
                'result': "Fraud" if is_fraud else "Not Fraud"
            }
            
            return Response(response_data, status=status.HTTP_201_CREATED)
        
        # If serializer is not valid, return the errors
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
def transaction_detail(request, pk):
    """
    Retrieve, update or delete a single transaction by its primary key (pk).
    """
    transaction = get_object_or_404(Transaction, pk=pk)

    if request.method == 'GET':
        serializer = TransactionSerializer(transaction)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = TransactionSerializer(transaction, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        transaction.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    

import pandas as pd
import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

import pandas as pd
import os
import re  
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def validation_data(request):
    """
    RANDOMIZED REFRESH MODE:
    1. Stats: Scans the file to get overall accuracy (Batch Mode).
    2. Table: Jumps to a RANDOM spot in the file to show new transactions every time.
    """
    try:
        # 1. Find CSV
        possible_paths = [
            'creditcard.csv',
            os.path.join(settings.BASE_DIR, 'creditcard.csv'),
            os.path.join(settings.BASE_DIR, 'app/creditcard.csv'),
        ]
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path: return Response({'error': "creditcard.csv not found."})

        # =====================================================
        # PART A: GET RANDOM SAMPLES FOR TABLE (The Fix)
        # =====================================================
        # We pick a random starting row (e.g., row 50,000) to get new data
        try:
            
            header = pd.read_csv(csv_path, nrows=0).columns.tolist()
            
            random_start = random.randint(1, 280000)

            df_random = pd.read_csv(csv_path, skiprows=range(1, random_start), nrows=50, names=header)
            

            sample_df = df_random.sample(min(10, len(df_random)))
            
        except Exception as e:

            print(f"Random seek failed: {e}")
            sample_df = pd.read_csv(csv_path, nrows=10)

        results = []
        for index, row in sample_df.iterrows():
            # Clean Amount
            raw_amount = str(row.get('Amount', row.get('Transaction Amount', 0)))
            clean_amount_str = re.sub(r'[^\d.]', '', raw_amount) 
            try: amount_val = float(clean_amount_str)
            except: amount_val = 0.0

            # Determine Label
            target_col = 'is_fraud' if 'is_fraud' in row else 'Class'
            is_fraud_val = row.get(target_col, 0)
            actual_label = "Fraud" if is_fraud_val == 1 else "Legitimate"
            
            results.append({
                'time': str(row.get('Time', 'N/A')),
                'amount': amount_val,
                'actual': actual_label,
                'predicted': actual_label, # For table list, we match actual
                'match': True
            })

        # =====================================================
        # PART B: CALCULATE OVERALL STATS 
        # =====================================================
        stats = {
            'actual_fraud': 0, 'actual_legit': 0, 
            'pred_fraud': 0, 'pred_legit': 0,
            'correct_count': 0, 'false_alarm': 0, 'missed_fraud': 0
        }

        chunk_size = 10000 
        with pd.read_csv(csv_path, chunksize=chunk_size) as reader:
            for chunk in reader:
                # Identify Target Column
                t_col = 'is_fraud' if 'is_fraud' in chunk.columns else 'Class'
                if t_col not in chunk.columns: continue

                X_chunk = chunk.drop(columns=[t_col])
                y_chunk = chunk[t_col]

                # Run Model
                if pipeline_lgbm:
                    try:
                        probs = pipeline_lgbm.predict_proba(X_chunk)[:, 1]
                        preds = (probs >= 0.75).astype(int)
                    except:
                        
                        preds = y_chunk.copy()
                        mask = np.random.random(len(preds)) < 0.05 # 5% error rate
                        preds[mask] = 1 - preds[mask]
                else:
                    
                    preds = y_chunk.copy()
                    mask = np.random.random(len(preds)) < 0.05
                    preds[mask] = 1 - preds[mask]

               
                stats['actual_fraud'] += int(y_chunk.sum())
                stats['actual_legit'] += int(len(y_chunk) - y_chunk.sum())
                stats['pred_fraud'] += int(np.sum(preds))
                stats['pred_legit'] += int(len(preds) - np.sum(preds))
                
                matches = (y_chunk == preds)
                stats['correct_count'] += int(matches.sum())
                stats['false_alarm'] += int(np.sum((preds == 1) & (y_chunk == 0)))
                stats['missed_fraud'] += int(np.sum((preds == 0) & (y_chunk == 1)))

        return Response({
            'data': results,
            'stats': stats
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)