import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screens/splash_screen.dart';
import 'screens/welcome_screen.dart';
import 'screens/prediction_screen.dart';
import 'utils/app_theme.dart';
import 'providers/theme_provider.dart';

void main() async {
  // Ensure Flutter is initialized
  WidgetsFlutterBinding.ensureInitialized();

  // Load theme preferences before building the app
  final prefs = await SharedPreferences.getInstance();
  final isDarkMode = prefs.getBool('isDarkMode') ?? false;

  runApp(
    ChangeNotifierProvider(
      create: (_) => ThemeProvider(initialDarkMode: isDarkMode),
      child: HydroponicApp(),
    ),
  );
}

class HydroponicApp extends StatelessWidget {
  const HydroponicApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Get the theme provider to listen for changes
    final themeProvider = Provider.of<ThemeProvider>(context);

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'FarmSmart - Hydroponic pH Predictor',
      // Set the light theme
      theme: AppTheme.lightTheme,
      // Set the dark theme
      darkTheme: AppTheme.darkTheme,
      // Set the theme mode based on the provider state
      themeMode: themeProvider.isDarkMode ? ThemeMode.dark : ThemeMode.light,
      initialRoute: '/',
      routes: {
        '/': (context) => SplashScreen(),
        '/welcome': (context) => WelcomeScreen(),
        '/predict': (context) => PredictionScreen(),
      },
    );
  }
}
