import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/theme_provider.dart';
import '../utils/app_theme.dart';

class ThemeToggle extends StatelessWidget {
  final bool isSmall;

  const ThemeToggle({super.key, this.isSmall = false});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDark = themeProvider.isDarkMode;

    return GestureDetector(
      onTap: () {
        themeProvider.toggleTheme();
      },
      child: Container(
        padding: EdgeInsets.all(isSmall ? 6 : 8),
        decoration: BoxDecoration(
          color:
              Theme.of(context).brightness == Brightness.dark
                  ? Colors.grey.shade800
                  : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isDark ? Icons.wb_sunny_outlined : Icons.dark_mode_outlined,
              size: isSmall ? 18 : 22,
              color: isDark ? Colors.orange : Colors.indigo,
            ),
            if (!isSmall) SizedBox(width: 6),
            if (!isSmall)
              Text(
                isDark ? "Light Mode" : "Dark Mode",
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  color:
                      isDark
                          ? AppColors.textPrimaryColorDark
                          : AppColors.textPrimaryColor,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
