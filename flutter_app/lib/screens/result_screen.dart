import 'dart:io';

import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final String imagePath;
  final String label;
  final double confidence;

  const ResultScreen({
    super.key,
    required this.imagePath,
    required this.label,
    required this.confidence,
  });

  @override
  Widget build(BuildContext context) {
    final triage = _triage(label, confidence);

    return Scaffold(
      appBar: AppBar(title: const Text('AI Result')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.file(File(imagePath), fit: BoxFit.cover),
              ),
            ),
            const SizedBox(height: 16),
            Card(
              color: triage.color.withOpacity(0.12),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    Icon(Icons.circle, color: triage.color, size: 48),
                    const SizedBox(height: 8),
                    Text(triage.title, style: Theme.of(context).textTheme.headlineSmall),
                    const SizedBox(height: 8),
                    Text('Model output: $label'),
                    Text('Confidence: ${(confidence * 100).toStringAsFixed(1)}%'),
                    const SizedBox(height: 8),
                    Text(triage.description, textAlign: TextAlign.center),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  _TriageState _triage(String label, double confidence) {
    if (label.toLowerCase().contains('normal') && confidence > 0.70) {
      return const _TriageState(
        title: 'Green — Healthy',
        description: 'No obvious high-risk pattern detected. Continue routine screening.',
        color: Colors.green,
      );
    }
    if (confidence < 0.70) {
      return const _TriageState(
        title: 'Yellow — Risk',
        description: 'Image suggests uncertainty or moderate risk. Repeat scan or consult a clinician.',
        color: Colors.orange,
      );
    }
    return const _TriageState(
      title: 'Red — Refer doctor',
      description: 'High-risk screening result. Recommend professional eye evaluation.',
      color: Colors.red,
    );
  }
}

class _TriageState {
  final String title;
  final String description;
  final Color color;

  const _TriageState({required this.title, required this.description, required this.color});
}
