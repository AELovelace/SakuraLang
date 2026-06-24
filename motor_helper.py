import time
from lovensepy import ServerClient

class MotorHelper:
    def __init__(self, developer_key, app_key):
        """
        Initialize the Lovense Cloud Client
        """
        print("🌸 Initializing Lovense Cloud Connection...")
        self.client = ServerClient(developer_key, app_key)
        
    def set_vibration(self, user_id, intensity, duration=2):
        """
        Send a vibration command to a specific user's toy.
        
        :param user_id: The UID obtained from the QR Code login
        :param intensity: Intensity level (0 to 20)
        :param duration: How long the vibration lasts in seconds
        """
        # Clamp intensity between 0 and 20
        intensity = max(0, min(20, int(intensity)))
        action = f"Vibrate:{intensity}"
        
        print(f"💖 Sending vibration: Level {intensity} for {duration}s to User {user_id}")
        
        try:
            self.client.function_request({
                "action": action,
                "timeSec": duration,
                "uid": user_id
            })
        except Exception as e:
            print(f"⚠️ Error sending vibration: {e}")

    def map_heat_to_intensity(self, heat_level):
        """
        Converts a 'Heat' level (0-100) to a Lovense Intensity (0-20)
        """
        # Simple linear mapping: 100 heat = 20 intensity
        return int((heat_level / 100) * 20)

    def stop(self, user_id):
        """
        Stops the toy by setting intensity to 0
        """
        self.set_vibration(user_id, 0, 0.1)
