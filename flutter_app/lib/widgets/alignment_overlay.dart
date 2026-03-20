import 'package:flutter/material.dart';

class AlignmentOverlay extends StatelessWidget {
  const AlignmentOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(
        painter: _AlignmentPainter(),
        child: Container(),
      ),
    );
  }
}

class _AlignmentPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final darkPaint = Paint()..color = Colors.black.withOpacity(0.35);
    final framePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.shortestSide * 0.32;

    final overlayPath = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final cutoutPath = Path()..addOval(Rect.fromCircle(center: center, radius: radius));
    final finalPath = Path.combine(PathOperation.difference, overlayPath, cutoutPath);

    canvas.drawPath(finalPath, darkPaint);
    canvas.drawCircle(center, radius, framePaint);

    final guidePaint = Paint()
      ..color = Colors.white70
      ..strokeWidth = 1.5;
    canvas.drawLine(Offset(center.dx - radius, center.dy), Offset(center.dx + radius, center.dy), guidePaint);
    canvas.drawLine(Offset(center.dx, center.dy - radius), Offset(center.dx, center.dy + radius), guidePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
