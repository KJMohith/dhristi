class ScanRecord {
  final String imagePath;
  final String label;
  final double confidence;
  final DateTime createdAt;

  ScanRecord({
    required this.imagePath,
    required this.label,
    required this.confidence,
    required this.createdAt,
  });

  Map<String, dynamic> toJson() => {
        'imagePath': imagePath,
        'label': label,
        'confidence': confidence,
        'createdAt': createdAt.toIso8601String(),
      };

  factory ScanRecord.fromJson(Map<String, dynamic> json) => ScanRecord(
        imagePath: json['imagePath'] as String,
        label: json['label'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        createdAt: DateTime.parse(json['createdAt'] as String),
      );
}
