"""
Workers package for background task execution.

Uses Taskiq with Redis Streams for reliable async task processing.
The broker is defined in workers.taskiq_app and tasks are registered
in the workers.tasks subpackage.
"""
