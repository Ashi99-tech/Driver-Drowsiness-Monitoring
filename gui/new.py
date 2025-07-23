def send_to_arduino(command):
    if arduino is None:
        print("[ERROR] Arduino serial not initialized")
        return
    try:
        arduino.write((command + '\n').encode())
        arduino.flush()
        print("[DEBUG] Sent to Arduino:", command)
    except Exception as e:
        print("[ERROR] Failed to send:", e)
