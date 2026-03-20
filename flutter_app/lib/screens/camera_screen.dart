import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../models.dart';
import '../services/history_service.dart';
import '../services/quality_service.dart';
import '../services/tflite_service.dart';
import '../widgets/alignment_overlay.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final QualityService _qualityService = QualityService();
  final TfliteService _tfliteService = TfliteService();
  CameraController? _controller;
  bool _isBusy = false;
  String? _message;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final backCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        backCamera,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();
      await _tfliteService.loadModel();
      if (!mounted) return;
      setState(() => _controller = controller);
    } catch (error) {
      if (!mounted) return;
      setState(() => _message = 'Camera initialization failed: $error');
    }
  }

  Future<void> _captureAndAnalyze() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _isBusy) return;

    setState(() {
      _isBusy = true;
      _message = null;
    });

    try {
      final tempDir = await getTemporaryDirectory();
      final rawCapture = await controller.takePicture();
      final savedPath = path.join(tempDir.path, 'drishti_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await File(rawCapture.path).copy(savedPath);

      final quality = await _qualityService.assessImage(savedPath);
      if (!quality.isGood) {
        setState(() {
          _message = 'Rejected image: ${quality.reasons.join(', ')}';
          _isBusy = false;
        });
        return;
      }

      final result = await _tfliteService.predict(savedPath);
      final record = ScanRecord(
        imagePath: savedPath,
        label: result.label,
        confidence: result.confidence,
        createdAt: DateTime.now(),
      );
      if (!mounted) return;
      await context.read<HistoryService>().addRecord(record);

      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            imagePath: savedPath,
            label: result.label,
            confidence: result.confidence,
          ),
        ),
      );
    } catch (error) {
      setState(() => _message = 'Scan failed: $error');
    } finally {
      if (mounted) {
        setState(() => _isBusy = false);
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;

    return Scaffold(
      appBar: AppBar(title: const Text('DRISHTI Capture')),
      body: Column(
        children: [
          Expanded(
            child: controller == null || !controller.value.isInitialized
                ? Center(child: Text(_message ?? 'Initializing camera...'))
                : Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(controller),
                      const AlignmentOverlay(),
                      Positioned(
                        left: 16,
                        right: 16,
                        bottom: 24,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Padding(
                            padding: EdgeInsets.all(12),
                            child: Text(
                              'Align the fundus inside the circle, keep the image sharp, then capture.',
                              style: TextStyle(color: Colors.white),
                              textAlign: TextAlign.center,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
          if (_message != null)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Text(_message!, style: const TextStyle(color: Colors.redAccent)),
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: FilledButton.icon(
              onPressed: _isBusy ? null : _captureAndAnalyze,
              icon: _isBusy
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.camera),
              label: Text(_isBusy ? 'Analyzing...' : 'Capture & Analyze'),
            ),
          ),
        ],
      ),
    );
  }
}
