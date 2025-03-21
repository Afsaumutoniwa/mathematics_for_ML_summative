import 'package:flutter/material.dart';

class AppColors {
  // Light theme colors
  static const primaryColor = Color(0xFF4CAF50);
  static const accentColor = Color(0xFF8BC34A);
  static const backgroundColor = Color(0xFFF5F9F6);
  static const textPrimaryColor = Color(0xFF2E3A59);
  static const textSecondaryColor = Color(0xFF7D8FAB);

  // Dark theme colors
  static const primaryColorDark = Color(0xFF2E7D32);
  static const accentColorDark = Color(0xFF558B2F);
  static const backgroundColorDark = Color(0xFF121212);
  static const cardColorDark = Color(0xFF1E1E1E);
  static const textPrimaryColorDark = Color(0xFFE0E0E0);
  static const textSecondaryColorDark = Color(0xFFAAAAAA);
}

class AppTheme {
  static final ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    primaryColor: AppColors.primaryColor,
    scaffoldBackgroundColor: AppColors.backgroundColor,
    cardColor: Colors.white,
    dividerColor: Colors.grey.shade300,
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.primaryColor,
        foregroundColor: Colors.white,
        padding: EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: Colors.white,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.grey.shade300),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.grey.shade300),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: AppColors.primaryColor, width: 2),
      ),
      contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 16),
    ),
    textTheme: TextTheme(
      headlineMedium: TextStyle(
        color: AppColors.textPrimaryColor,
        fontWeight: FontWeight.bold,
      ),
      bodyLarge: TextStyle(color: AppColors.textPrimaryColor),
      bodyMedium: TextStyle(color: AppColors.textSecondaryColor),
    ),
    iconTheme: IconThemeData(color: AppColors.primaryColor),
  );

  static final ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    primaryColor: AppColors.primaryColorDark,
    scaffoldBackgroundColor: AppColors.backgroundColorDark,
    cardColor: AppColors.cardColorDark,
    dividerColor: Colors.grey.shade800,
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.primaryColorDark,
        foregroundColor: Colors.white,
        padding: EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.cardColorDark,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.grey.shade700),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.grey.shade700),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: AppColors.primaryColorDark, width: 2),
      ),
      contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      labelStyle: TextStyle(color: AppColors.textSecondaryColorDark),
      hintStyle: TextStyle(
        color: AppColors.textSecondaryColorDark.withOpacity(0.7),
      ),
    ),
    textTheme: TextTheme(
      headlineMedium: TextStyle(
        color: AppColors.textPrimaryColorDark,
        fontWeight: FontWeight.bold,
      ),
      bodyLarge: TextStyle(color: AppColors.textPrimaryColorDark),
      bodyMedium: TextStyle(color: AppColors.textSecondaryColorDark),
    ),
    iconTheme: IconThemeData(color: AppColors.primaryColorDark),
    dropdownMenuTheme: DropdownMenuThemeData(
      inputDecorationTheme: InputDecorationTheme(
        fillColor: AppColors.cardColorDark,
      ),
    ),
  );
}
