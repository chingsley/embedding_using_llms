import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Logger {
    private String logFile;

    public Logger(String logFile) {
        this.logFile = logFile;
    }

    // Write a log message to the file
    public void log(String message) {
        try (FileWriter writer = new FileWriter(logFile, true)) {
            writer.write(message + "\n");
            System.out.println("Logged: " + message);
        } catch (IOException e) {
            System.out.println("Error writing to log file: " + e.getMessage());
        }
    }

    // Read all logs from the file
    public void readLogs() {
        try {
            List<String> logs = Files.readAllLines(Paths.get(logFile));
            System.out.println("Logs:");
            logs.forEach(System.out::println);
        } catch (IOException e) {
            System.out.println("Error reading log file: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        Logger logger = new Logger("app.log");
        logger.log("Application started.");
        logger.log("User logged in.");
        logger.readLogs();
    }
}