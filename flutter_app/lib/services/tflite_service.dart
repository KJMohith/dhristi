import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class InferenceResult {
  final String label;
  final double confidence;
  final List<double> scores;

  InferenceResult({
    required this.label,
    required this.confidence,
    required this.scores,
  });
}

class TfliteService {
  static const _labels = ['Glaucoma', 'Normal'];
  Interpreter? _interpreter;

  Future<void> loadModel() async {
    try {
      _interpreter ??= await Interpreter.fromAsset('assets/models/drishti_model.tflite');
    } catch (error) {
      throw Exception(
        'Unable to load the bundled AI model. '
        'Make sure flutter_app/assets/models/drishti_model.tflite is packaged before deployment. '
        'Original error: $error',
      );
    }
  }

  Future<InferenceResult> predict(String imagePath) async {
    await loadModel();
    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Could not decode image for inference.');
    }

    final resized = img.copyResize(decoded, width: 224, height: 224);
    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resized.getPixel(x, y);
            return [
              pixel.r / 255.0,
              pixel.g / 255.0,
              pixel.b / 255.0,
            ];
          },
        ),
      ),
    );

    final output = [List<double>.filled(_labels.length, 0)];
    _interpreter!.run(input, output);

    final scores = output.first;
    final maxScore = scores.reduce(math.max);
    final maxIndex = scores.indexOf(maxScore);

    return InferenceResult(
      label: _labels[maxIndex],
      confidence: maxScore,
      scores: scores,
    );
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
