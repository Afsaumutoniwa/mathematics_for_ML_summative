import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class PredictionResultModal {
  static void show(
    BuildContext context, {
    required String predictionResult,
    required String suggestion,
  }) {
    showDialog(
      context: context,
      builder:
          (context) => Dialog(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24),
            ),
            elevation: 8,
            backgroundColor: Colors.transparent,
            child: PredictionResultContent(
              predictionResult: predictionResult,
              suggestion: suggestion,
            ),
          ),
    );
  }
}

class PredictionResultContent extends StatelessWidget {
  final String predictionResult;
  final String suggestion;

  const PredictionResultContent({
    super.key,
    required this.predictionResult,
    required this.suggestion,
  });

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Theme.of(context).brightness == Brightness.dark;
    final isOptimal = suggestion.contains("optimal");
    final isLow = suggestion.contains("low");

    final statusColor =
        isOptimal
            ? Theme.of(context).primaryColor
            : isLow
            ? Color(0xFFFF9800)
            : Color(0xFF2196F3);

    final phValue = predictionResult.replaceAll("Predicted pH: ", "");

    return Stack(
      clipBehavior: Clip.none,
      children: [
        // Main card
        Container(
          padding: EdgeInsets.only(top: 60, left: 24, right: 24, bottom: 24),
          decoration: BoxDecoration(
            color: Theme.of(context).cardColor,
            borderRadius: BorderRadius.circular(24),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(height: 16),
              Text(
                "pH Result",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w600,
                  color:
                      isDarkMode
                          ? AppColors.textPrimaryColorDark
                          : Colors.black87,
                ),
              ),
              SizedBox(height: 40),
              // Big pH value
              Text(
                phValue,
                style: TextStyle(
                  fontSize: 56,
                  fontWeight: FontWeight.bold,
                  color: statusColor,
                ),
              ),
              SizedBox(height: 8),
              // Status tag
              Container(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: statusColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  isOptimal
                      ? "Optimal"
                      : isLow
                      ? "Too Low"
                      : "Too High",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: statusColor,
                  ),
                ),
              ),
              SizedBox(height: 24),
              Divider(
                color: isDarkMode ? Colors.grey.shade800 : Colors.grey.shade300,
              ),
              SizedBox(height: 16),
              // Recommendation section
              Row(
                children: [
                  Icon(Icons.lightbulb_outline, color: Colors.amber),
                  SizedBox(width: 12),
                  Text(
                    "Recommendation",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color:
                          isDarkMode
                              ? AppColors.textPrimaryColorDark
                              : Colors.black87,
                    ),
                  ),
                ],
              ),
              SizedBox(height: 12),
              Text(
                suggestion,
                style: TextStyle(
                  fontSize: 16,
                  color:
                      isDarkMode
                          ? AppColors.textSecondaryColorDark
                          : Colors.black54,
                  height: 1.4,
                ),
              ),
              SizedBox(height: 24),
              // Action buttons
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => Navigator.of(context).pop(),
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: statusColor),
                        padding: EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: Text(
                        "Close",
                        style: TextStyle(
                          color: statusColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                        // Optional: add action for "Save" or "Apply" here
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: statusColor,
                        padding: EdgeInsets.symmetric(vertical: 12),
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: Text(
                        "New Prediction",
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),

        // Circular icon at the top
        Positioned(
          top: -40,
          left: 0,
          right: 0,
          child: Center(
            child: Container(
              height: 80,
              width: 80,
              decoration: BoxDecoration(
                color: Theme.of(context).cardColor,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 8,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: Center(
                child: Container(
                  height: 70,
                  width: 70,
                  decoration: BoxDecoration(
                    color: statusColor,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    isOptimal
                        ? Icons.check_circle_outline
                        : isLow
                        ? Icons.arrow_downward
                        : Icons.arrow_upward,
                    size: 40,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
