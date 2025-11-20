# app/serializers.py

from rest_framework import serializers
from .models import Transaction

class TransactionSerializer(serializers.ModelSerializer):
    
    # 1. ADD THIS LINE: Define the new "predicted" field
    predicted = serializers.SerializerMethodField()

    class Meta:
        model = Transaction
        # 2. 'fields' is still '__all__'
        fields = '__all__' 
        # 3. REMOVE the 'read_only_fields' line that was causing the error
    
    # 4. ADD THIS FUNCTION: This tells the 'predicted' field
    #    what value to show based on the 'is_fraud' column.
    def get_predicted(self, obj):
        if obj.is_fraud:
            return "Fraud"
        return "Not Fraud"