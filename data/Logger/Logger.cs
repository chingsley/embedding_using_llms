using System;
using System.IO;
using System.Collections.Generic;

public class Logger {
    private string logFile;

    public Logger(string logFile) {
        this.logFile = logFile;
    }

    // Write a log message to the file
    public void Log(string message) {
        try {
            using (StreamWriter writer = new StreamWriter(logFile, true)) {
                writer.WriteLine(message);
                Console.WriteLine("Logged: " + message);
            }
        } catch (Exception e) {
            Console.WriteLine("Error writing to log file: " + e.Message);
        }
    }

    // Read all logs from the file
    public void ReadLogs() {
        try {
            string[] logs = File.ReadAllLines(logFile);
            Console.WriteLine("Logs:");
            foreach (string log in logs) {
                Console.WriteLine(log);
            }
        } catch (Exception e) {
            Console.WriteLine("Error reading log file: " + e.Message);
        }
    }

    public static void Main(string[] args) {
        Logger logger = new Logger("app.log");
        logger.Log("Application started.");
        logger.Log("User logged in.");
        logger.ReadLogs();
    }
}