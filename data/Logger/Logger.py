class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    # Write a log message to the file
    def log(self, message):
        try:
            with open(self.log_file, "a") as file:
                file.write(message + "\n")
            print(f"Logged: {message}")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    # Read all logs from the file
    def read_logs(self):
        try:
            with open(self.log_file, "r") as file:
                logs = file.readlines()
            print("Logs:")
            for log in logs:
                print(log, end="")
        except Exception as e:
            print(f"Error reading log file: {e}")

# Testing the class
if __name__ == "__main__":
    logger = Logger("app.log")
    logger.log("Application started.")
    logger.log("User logged in.")
    logger.read_logs()