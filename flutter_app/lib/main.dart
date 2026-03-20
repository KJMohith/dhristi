import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/camera_screen.dart';
import 'screens/history_screen.dart';
import 'services/history_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final historyService = HistoryService();
  await historyService.init();

  runApp(
    ChangeNotifierProvider.value(
      value: historyService,
      child: const DrishtiApp(),
    ),
  );
}

class DrishtiApp extends StatelessWidget {
  const DrishtiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DRISHTI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const MainNavigationShell(),
    );
  }
}

class MainNavigationShell extends StatefulWidget {
  const MainNavigationShell({super.key});

  @override
  State<MainNavigationShell> createState() => _MainNavigationShellState();
}

class _MainNavigationShellState extends State<MainNavigationShell> {
  int _index = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      const CameraScreen(),
      const HistoryScreen(),
    ];

    return Scaffold(
      body: pages[_index],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (value) => setState(() => _index = value),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.camera_alt_outlined), label: 'Scan'),
          NavigationDestination(icon: Icon(Icons.history_outlined), label: 'History'),
        ],
      ),
    );
  }
}
