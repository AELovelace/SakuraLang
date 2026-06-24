"""
Test script to verify Lovense Cloud API connection.
Run this to make sure the toy buzzes!
"""
from motor_helper import MotorHelper

# --- CONFIGURATION ---
# Replace these with your actual keys!
DEVELOPER_KEY = "YOUR_DEVELOPER_KEY_HERE"
APP_KEY = "YOUR_APP_KEY_HERE"
USER_ID = "USER_UID_FROM_QR_CODE"
# ---------------------

def main():
    # 1. Initialize the helper
    helper = MotorHelper(DEVELOPER_KEY, APP_KEY)
    
    print(f"🌸 Connected! Testing motor for User ID: {USER_ID}...")
    
    # 2. Test Vibration (Medium Intensity)
    print("🔥 Buzzing...")
    helper.set_vibration(USER_ID, intensity=10, duration=3)
    
    # Wait for it to finish
    import time
    time.sleep(3)
    
    # 3. Stop
    print("✨ Stopping...")
    helper.stop(USER_ID)

if __name__ == "__main__":
    main()
