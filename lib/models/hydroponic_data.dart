class HydroponicInput {
  final double soilEc;
  final double nitrogen;
  final double phosphorus;
  final double potassium;
  final double moisture;
  final double temperature;
  final int crop;

  HydroponicInput({
    required this.soilEc,
    required this.nitrogen,
    required this.phosphorus,
    required this.potassium,
    required this.moisture,
    required this.temperature,
    required this.crop,
  });

  Map<String, dynamic> toJson() {
    return {
      "soil_ec": soilEc,
      "nitrogen": nitrogen,
      "phosphorus": phosphorus,
      "potassium": potassium,
      "moisture": moisture,
      "temperature": temperature,
      "crop": crop,
    };
  }
}
