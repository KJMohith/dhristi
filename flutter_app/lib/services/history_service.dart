import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models.dart';

class HistoryService extends ChangeNotifier {
  static const _storageKey = 'drishti_scan_history';
  final List<ScanRecord> _records = [];
  SharedPreferences? _prefs;

  List<ScanRecord> get records => List.unmodifiable(_records.reversed);

  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    final raw = _prefs?.getStringList(_storageKey) ?? [];
    _records
      ..clear()
      ..addAll(raw.map((item) => ScanRecord.fromJson(jsonDecode(item) as Map<String, dynamic>)));
    notifyListeners();
  }

  Future<void> addRecord(ScanRecord record) async {
    _records.add(record);
    await _persist();
    notifyListeners();
  }

  Future<void> clear() async {
    _records.clear();
    await _persist();
    notifyListeners();
  }

  Future<void> _persist() async {
    await _prefs?.setStringList(
      _storageKey,
      _records.map((record) => jsonEncode(record.toJson())).toList(),
    );
  }
}
