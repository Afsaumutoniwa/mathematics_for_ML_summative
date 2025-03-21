import 'package:flutter/material.dart';
import '../utils/app_theme.dart';
import '../widgets/theme_toggle.dart';

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      body: Column(
        children: [
          Container(
            height: 100,
            width: double.infinity,
            color: Theme.of(context).primaryColor,
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [ThemeToggle()],
                ),
              ),
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    SizedBox(height: 10),
                    Image.asset("assets/logo.png", height: 200),
                    SizedBox(height: 30),
                    Text(
                      "🌱 At FarmSmart, we are dedicated to revolutionizing agriculture with technology and data-driven insights.\n\n\n"
                      "💧 Our primary focus is hydroponic farming, which allows crops to grow without soil, using nutrient-rich water solutions to maximize yield and efficiency.\n\n\n"
                      "📊 This app helps you analyze and optimize pH levels for healthier crops and improved production.\n",
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w500,
                        color:
                            isDarkMode
                                ? AppColors.textPrimaryColorDark
                                : Colors.black87,
                      ),
                    ),
                    SizedBox(height: 30),
                    ElevatedButton(
                      onPressed:
                          () => Navigator.pushReplacementNamed(
                            context,
                            '/predict',
                          ),
                      child: Text("Proceed to Prediction"),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
