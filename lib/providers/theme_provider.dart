import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeProvider with ChangeNotifier {
  bool _isDarkMode = false;
  bool get isDarkMode => _isDarkMode;

  ThemeProvider({bool initialDarkMode = false}) {
    _isDarkMode = initialDarkMode;
  }

  // Toggle between dark and light theme
  void toggleTheme() async {
    _isDarkMode = !_isDarkMode;

    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setBool('isDarkMode', _isDarkMode);

    notifyListeners();
  }
}
