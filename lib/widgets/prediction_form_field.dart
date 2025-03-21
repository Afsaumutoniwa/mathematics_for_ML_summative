import 'package:flutter/material.dart';

class PredictionFormField extends StatelessWidget {
  final TextEditingController controller;
  final String labelText;
  final String? Function(String?)? validator;
  final IconData? icon;

  const PredictionFormField({
    super.key,
    required this.controller,
    required this.labelText,
    this.validator,
    this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10.0),
      child: TextFormField(
        controller: controller,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(
          labelText: labelText,
          border: OutlineInputBorder(),
          prefixIcon: icon != null ? Icon(icon) : null,
        ),
        validator: validator,
      ),
    );
  }
}
