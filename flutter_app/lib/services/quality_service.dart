import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;

class QualityResult {
  final bool isGood;
  final double blurScore;
  final double brightnessScore;
  final double centerScore;
  final List<String> reasons;

  QualityResult({
    required this.isGood,
    required this.blurScore,
    required this.brightnessScore,
    required this.centerScore,
    required this.reasons,
  });
}

class QualityService {
  Future<QualityResult> assessImage(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) {
      return QualityResult(
        isGood: false,
        blurScore: 0,
        brightnessScore: 0,
        centerScore: 0,
        reasons: ['Unable to decode image'],
      );
    }

    final blur = _laplacianVariance(image);
    final brightness = _averageBrightness(image);
    final center = _centerScore(image);

    final reasons = <String>[];
    if (blur < 80) reasons.add('Image is blurry');
    if (brightness < 45 || brightness > 220) reasons.add('Brightness is not ideal');
    if (center < 0.45) reasons.add('Retina is not centered');

    return QualityResult(
      isGood: reasons.isEmpty,
      blurScore: blur,
      brightnessScore: brightness,
      centerScore: center,
      reasons: reasons,
    );
  }

  double _averageBrightness(img.Image image) {
    double total = 0;
    final pixels = image.width * image.height;
    for (final pixel in image) {
      total += img.getLuminance(pixel);
    }
    return total / pixels;
  }

  double _centerScore(img.Image image) {
    double weightedX = 0;
    double weightedY = 0;
    double total = 0;

    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final luminance = img.getLuminance(pixel);
        if (luminance > 200) {
          weightedX += x * luminance;
          weightedY += y * luminance;
          total += luminance;
        }
      }
    }

    if (total == 0) return 0;

    final cx = weightedX / total;
    final cy = weightedY / total;
    final dx = cx - image.width / 2;
    final dy = cy - image.height / 2;
    final distance = math.sqrt(dx * dx + dy * dy);
    final maxDistance = math.sqrt(
      math.pow(image.width / 2, 2) + math.pow(image.height / 2, 2),
    );
    return 1 - (distance / maxDistance).clamp(0, 1);
  }

  double _laplacianVariance(img.Image image) {
    final grayscale = img.grayscale(image);
    final values = <int>[];

    for (var y = 1; y < grayscale.height - 1; y++) {
      for (var x = 1; x < grayscale.width - 1; x++) {
        final center = img.getLuminance(grayscale.getPixel(x, y));
        final up = img.getLuminance(grayscale.getPixel(x, y - 1));
        final down = img.getLuminance(grayscale.getPixel(x, y + 1));
        final left = img.getLuminance(grayscale.getPixel(x - 1, y));
        final right = img.getLuminance(grayscale.getPixel(x + 1, y));
        final laplacian = (4 * center - up - down - left - right).abs();
        values.add(laplacian);
      }
    }

    if (values.isEmpty) return 0;
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance = values
            .map((value) => math.pow(value - mean, 2))
            .reduce((a, b) => a + b) /
        values.length;
    return variance.toDouble();
  }
}
