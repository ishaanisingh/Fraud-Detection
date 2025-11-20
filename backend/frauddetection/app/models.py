# app/models.py

from django.db import models

class Transaction(models.Model):
    # --- Customer Information ---
    customer_id = models.IntegerField(primary_key=True) 
    name = models.CharField(max_length=100)
    surname = models.CharField(max_length=100)
    gender = models.CharField(max_length=10) 
    birthdate = models.DateField()
    country_of_residence = models.CharField(max_length=100)
    age = models.IntegerField()
    job = models.CharField(max_length=100)

    # --- Card and Bank Information ---
    cc_num = models.CharField(max_length=20) 
    type_of_card = models.CharField(max_length=50)
    bank = models.CharField(max_length=100)

    # --- Transaction Details ---
    transaction_id = models.CharField(max_length=50, unique=True)
    trans_num = models.CharField(max_length=50) 
    transaction_amount = models.DecimalField(max_digits=10, decimal_places=2)
    
    # --- !! CRITICAL FIX !! ---
    # Your model pipeline was trained with 'Amount' as a categorical string,
    # so this *must* be a CharField, not a DecimalField.
    amount = models.CharField(max_length=50) 
    
    type_of_transaction = models.CharField(max_length=50)
    entry_mode = models.CharField(max_length=50)
    
    # This will be updated by your prediction model
    is_fraud = models.BooleanField(default=False) 

    # --- Merchant Information ---
    merchant_name = models.CharField(max_length=100)
    category = models.CharField(max_length=100)
    merchant_group = models.CharField(max_length=100)
    country_of_transaction = models.CharField(max_length=100)
    merch_lat = models.FloatField() 
    merch_long = models.FloatField()

    # --- Time and Date Information ---
    date = models.DateField()
    day_of_week = models.CharField(max_length=20)
    time = models.TimeField()
    trans_date_trans_time = models.DateTimeField() 
    unix_time = models.BigIntegerField()

    # --- Location Information ---
    shipping_address = models.CharField(max_length=255)
    state = models.CharField(max_length=50)
    zip = models.CharField(max_length=10) 
    lat = models.FloatField() 
    long = models.FloatField() 
    city_pop = models.IntegerField()

    def __str__(self):
        return f"Transaction {self.transaction_id} - {'FRAUD' if self.is_fraud else 'Not Fraud'}"