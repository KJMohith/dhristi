import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/history_service.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final history = context.watch<HistoryService>();
    final records = history.records;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        actions: [
          IconButton(
            onPressed: records.isEmpty ? null : history.clear,
            icon: const Icon(Icons.delete_outline),
          ),
        ],
      ),
      body: records.isEmpty
          ? const Center(child: Text('No offline scans yet.'))
          : ListView.separated(
              itemCount: records.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final record = records[index];
                return ListTile(
                  leading: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.file(File(record.imagePath), width: 56, height: 56, fit: BoxFit.cover),
                  ),
                  title: Text(record.label),
                  subtitle: Text(
                    '${(record.confidence * 100).toStringAsFixed(1)}% • ${record.createdAt.toLocal()}',
                  ),
                );
              },
            ),
    );
  }
}
