import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/hydroponic_data.dart';

class ApiService {
  final String baseUrl = "http://10.0.2.2:8000";

  Future<double> predictPH(HydroponicInput data) async {
    final url = Uri.parse("$baseUrl/predict");
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(data.toJson()),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body)["Predicted pH"];
    } else {
      throw Exception('Failed to predict pH');
    }
  }
}
