import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
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
  final ImagePicker _imagePicker = ImagePicker();
  CameraController? _controller;
  bool _isBusy = false;
  bool _isInitializing = true;
  String? _message;

  @override
  void initState() {
    super.initState();
    _initializeExperience();
  }

  Future<void> _initializeExperience() async {
    setState(() {
      _isInitializing = true;
      _message = null;
    });

    try {
      await _tfliteService.loadModel();
      await _initCamera();
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
        _message = 'Setup failed: $error';
      });
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw Exception(
          'No camera was detected on this device. You can still analyze a gallery photo below.',
        );
      }
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
      await _controller?.dispose();
      if (!mounted) return;
      setState(() {
        _controller = controller;
        _isInitializing = false;
      });
    } catch (error) {
      await _controller?.dispose();
      if (!mounted) return;
      setState(() {
        _controller = null;
        _isInitializing = false;
        _message = 'Camera unavailable: $error';
      });
    }
  }

  Future<void> _analyzeImage(String sourcePath) async {
    setState(() {
      _isBusy = true;
      _message = null;
    });

    try {
      final tempDir = await getTemporaryDirectory();
      final fileExtension = path.extension(sourcePath).isEmpty
          ? '.jpg'
          : path.extension(sourcePath);
      final savedPath = path.join(
        tempDir.path,
        'drishti_${DateTime.now().millisecondsSinceEpoch}$fileExtension',
      );
      await File(sourcePath).copy(savedPath);

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

  Future<void> _captureAndAnalyze() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _isBusy) return;

    final rawCapture = await controller.takePicture();
    await _analyzeImage(rawCapture.path);
  }

  Future<void> _pickFromGallery() async {
    if (_isBusy || _isInitializing) return;

    try {
      final pickedFile = await _imagePicker.pickImage(source: ImageSource.gallery);
      if (pickedFile == null) {
        return;
      }
      await _analyzeImage(pickedFile.path);
    } catch (error) {
      if (!mounted) return;
      setState(() => _message = 'Could not open the gallery: $error');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _tfliteService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final isCameraReady = controller != null && controller.value.isInitialized;

    return Scaffold(
      appBar: AppBar(title: const Text('DRISHTI Capture')),
      body: Column(
        children: [
          Expanded(
            child: isCameraReady
                ? Stack(
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
                  )
                : Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          if (_isInitializing) ...[
                            const CircularProgressIndicator(),
                            const SizedBox(height: 16),
                            const Text('Initializing camera and offline model...'),
                          ] else ...[
                            Icon(
                              Icons.photo_library_outlined,
                              size: 56,
                              color: Theme.of(context).colorScheme.primary,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              _message ?? 'Camera is unavailable, but gallery analysis is still ready.',
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 16),
                            FilledButton.icon(
                              onPressed: _initializeExperience,
                              icon: const Icon(Icons.refresh),
                              label: const Text('Retry setup'),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
          ),
          if (_message != null)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Text(_message!, style: const TextStyle(color: Colors.redAccent)),
            ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                FilledButton.icon(
                  onPressed: _isBusy || _isInitializing || !isCameraReady ? null : _captureAndAnalyze,
                  icon: _isBusy
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.camera),
                  label: Text(_isBusy ? 'Analyzing...' : 'Capture & Analyze'),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed: _isBusy || _isInitializing ? null : _pickFromGallery,
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Choose from Gallery'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
