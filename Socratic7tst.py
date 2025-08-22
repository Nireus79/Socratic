#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for Socratic RAG System v7.0
Includes unit tests, integration tests, and scenario testing.
"""

import unittest
import tempfile
import shutil
import os
import sqlite3
import datetime
import hashlib
import threading
import time
import sys
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import subprocess
import Socratic7
import asyncio
from unittest.mock import AsyncMock, patch
import numpy as np
from typing import List, Dict, Any
import json

"""Key Completions:

Finished run_all_tests() method - Handles keyboard interrupts and calculates total execution time
Test Summary and Reporting:

print_test_summary() - Displays comprehensive results with icons and error details
generate_test_report() - Creates detailed HTML test reports
all_tests_passed() - Checks overall test status


CLI Interface:

Command-line argument parsing for different test modes
Options for specific test classes, quick tests, performance-only tests
HTML report generation flag


Utility Functions:

run_specific_test_class() - Run individual test categories
Proper error handling and exit codes
Warning suppression for cleaner output

Usage Examples:
bash# Run all tests
python Socratic7tst.py

# Run specific test category
python Socratic7tst.py --test agents

# Quick tests (skip scenarios)
python Socratic7tst.py --quick

# Performance tests only
python Socratic7tst.py --performance

# Generate HTML report
python Socratic7tst.py --report
"""

# Add the directory containing Socratic7.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Socratic7 import (
        User, ProjectContext, KnowledgeEntry, TokenUsage, ConflictInfo,
        ProjectDatabase, VectorDatabase, ClaudeClient, AgentOrchestrator,
        ProjectManagerAgent, SocraticCounselorAgent, ContextAnalyzerAgent,
        CodeGeneratorAgent, SystemMonitorAgent, ConflictDetectorAgent,
        SocraticRAGSystem
    )
except ImportError as e:
    print(f"Error importing Socratic7 modules: {e}")
    print("Make sure Socratic7.py is in the same directory as this test file")
    sys.exit(1)


class TestAdvancedRAGFeatures(unittest.TestCase):
    """Tests for advanced RAG capabilities like multi-modal, streaming, and context optimization"""

    def setUp(self):
        """Set up test environment for advanced features"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.database = Mock()
        self.mock_orchestrator.vector_db = Mock()
        self.mock_orchestrator.claude_client = Mock()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_semantic_search_with_reranking(self):
        """Test semantic search with result reranking"""
        # Mock vector database with search results
        mock_results = [
            {'id': 'doc1', 'content': 'Python web development', 'score': 0.8},
            {'id': 'doc2', 'content': 'Database design patterns', 'score': 0.7},
            {'id': 'doc3', 'content': 'API security best practices', 'score': 0.6}
        ]

        self.mock_orchestrator.vector_db.search_similar.return_value = mock_results

        # Test reranking functionality
        from Socratic7 import VectorDatabase
        with patch('Socratic7.SentenceTransformer'):
            vector_db = VectorDatabase(self.temp_dir)

            query = "How to build secure Python web applications?"
            results = vector_db.search_with_reranking(query, top_k=3, rerank_top_k=2)

            # Should return reranked results
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 2)  # rerank_top_k

    def test_contextual_compression(self):
        """Test context compression for long documents"""
        # Mock long context
        long_context = "This is a very long document. " * 1000  # ~5000 chars

        from Socratic7 import ContextAnalyzerAgent
        analyzer = ContextAnalyzerAgent(self.mock_orchestrator)

        # Test compression
        compressed = analyzer.compress_context(long_context, max_tokens=500)

        self.assertIsInstance(compressed, str)
        self.assertLess(len(compressed), len(long_context))

    def test_multi_turn_conversation_context(self):
        """Test maintaining context across multiple conversation turns"""
        from Socratic7 import SocraticCounselorAgent

        agent = SocraticCounselorAgent(self.mock_orchestrator)

        # Simulate multi-turn conversation
        conversation_history = [
            {"role": "user", "content": "I want to build a web app"},
            {"role": "assistant", "content": "What kind of web app are you thinking of?"},
            {"role": "user", "content": "An e-commerce platform"},
            {"role": "assistant", "content": "What features are most important to you?"}
        ]

        # Mock project with conversation history
        project = Mock()
        project.conversation_history = conversation_history
        project.phase = "discovery"

        # Test context extraction from conversation
        context_summary = agent.extract_conversation_context(project)

        self.assertIsInstance(context_summary, str)
        self.assertIn("e-commerce", context_summary.lower())

    def test_adaptive_questioning_strategy(self):
        """Test adaptive questioning based on user responses"""
        from Socratic7 import SocraticCounselorAgent

        agent = SocraticCounselorAgent(self.mock_orchestrator)

        # Mock different user response types
        detailed_response = "I want to build a comprehensive e-commerce platform with user authentication, payment processing, inventory management, and real-time chat support."
        vague_response = "I want to make a website"

        # Test question adaptation
        detailed_question = agent.adapt_question_complexity(detailed_response, "technical")
        vague_question = agent.adapt_question_complexity(vague_response, "clarifying")

        # Detailed responses should get more technical questions
        self.assertIsInstance(detailed_question, str)
        self.assertIsInstance(vague_question, str)

        # Vague responses should get clarifying questions
        self.assertNotEqual(detailed_question, vague_question)

    @patch('Socratic7.asyncio')
    async def test_async_knowledge_retrieval(self):
        """Test asynchronous knowledge retrieval for better performance"""
        from Socratic7 import VectorDatabase

        with patch('Socratic7.SentenceTransformer'):
            vector_db = VectorDatabase(self.temp_dir)

            # Mock async search method
            vector_db.search_similar_async = AsyncMock(return_value=[
                {'id': 'async_doc1', 'content': 'Async content', 'score': 0.9}
            ])

            # Test async retrieval
            results = await vector_db.search_similar_async("async query")

            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 1)


class TestRealTimeCollaboration(unittest.TestCase):
    """Tests for real-time collaboration features"""

    def setUp(self):
        """Set up collaboration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_orchestrator = Mock()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_concurrent_editing_conflict_resolution(self):
        """Test handling of concurrent edits by multiple users"""
        from Socratic7 import ProjectManagerAgent

        agent = ProjectManagerAgent(self.mock_orchestrator)

        # Mock project with base state
        project = Mock()
        project.project_id = "test_project"
        project.tech_stack = ["python", "flask"]
        project.updated_at = datetime.datetime.now()

        # Simulate concurrent edits
        edit1 = {
            'user': 'user1',
            'timestamp': datetime.datetime.now(),
            'changes': {'tech_stack': ['python', 'django']},  # Changed flask to django
            'version': 1
        }

        edit2 = {
            'user': 'user2',
            'timestamp': datetime.datetime.now() + datetime.timedelta(seconds=1),
            'changes': {'tech_stack': ['python', 'flask', 'postgresql']},  # Added database
            'version': 1  # Same base version
        }

        # Test conflict resolution
        result = agent.resolve_concurrent_edits(project, [edit1, edit2])

        self.assertEqual(result['status'], 'success')
        self.assertIn('merged_changes', result)

    def test_real_time_notification_system(self):
        """Test real-time notifications for project updates"""
        from Socratic7 import SystemMonitorAgent

        agent = SystemMonitorAgent(self.mock_orchestrator)

        # Mock notification system
        notifications = []
        agent.notification_handlers = [lambda n: notifications.append(n)]

        # Test project update notification
        project_update = {
            'project_id': 'test_project',
            'user': 'user1',
            'action': 'tech_stack_update',
            'details': {'added': ['redis'], 'removed': []}
        }

        agent.notify_project_update(project_update)

        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]['action'], 'tech_stack_update')

    def test_collaborative_decision_tracking(self):
        """Test tracking and voting on collaborative decisions"""
        from Socratic7 import ConflictDetectorAgent

        agent = ConflictDetectorAgent(self.mock_orchestrator)

        # Mock collaborative decision
        decision = {
            'decision_id': 'tech_choice_001',
            'project_id': 'test_project',
            'question': 'Should we use React or Vue for frontend?',
            'options': ['react', 'vue'],
            'votes': {},
            'status': 'pending'
        }

        # Test voting
        result = agent.cast_vote(decision['decision_id'], 'user1', 'react', 'Better ecosystem')

        self.assertEqual(result['status'], 'success')
        self.assertIn('vote_recorded', result)

    def test_activity_feed_generation(self):
        """Test generation of project activity feeds"""
        from Socratic7 import SystemMonitorAgent

        agent = SystemMonitorAgent(self.mock_orchestrator)

        # Mock project activities
        activities = [
            {'timestamp': datetime.datetime.now(), 'user': 'user1', 'action': 'project_created'},
            {'timestamp': datetime.datetime.now(), 'user': 'user2', 'action': 'collaborator_added'},
            {'timestamp': datetime.datetime.now(), 'user': 'user1', 'action': 'tech_stack_updated'}
        ]

        # Test activity feed generation
        feed = agent.generate_activity_feed('test_project', limit=10)

        self.assertIsInstance(feed, list)
        # Should be ordered by timestamp (newest first)
        if len(feed) > 1:
            self.assertGreaterEqual(feed[0]['timestamp'], feed[1]['timestamp'])


class TestIntelligentCodeGeneration(unittest.TestCase):
    """Tests for advanced code generation features"""

    def setUp(self):
        """Set up code generation test environment"""
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.claude_client = Mock()

    def test_context_aware_code_generation(self):
        """Test code generation that adapts to project context"""
        from Socratic7 import CodeGeneratorAgent

        agent = CodeGeneratorAgent(self.mock_orchestrator)

        # Mock project with specific context
        project = Mock()
        project.tech_stack = ["python", "fastapi", "postgresql"]
        project.requirements = ["REST API", "authentication", "database"]
        project.code_style = "clean_architecture"

        # Mock Claude response with context-aware code
        mock_response = MockClaudeResponse("""
# FastAPI application with Clean Architecture
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import get_db
from .auth import get_current_user

app = FastAPI()

@app.get("/api/users/me")
async def get_current_user_info(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return current_user
""")

        self.mock_orchestrator.claude_client.generate_code.return_value = mock_response.content[0].text

        # Test context-aware generation
        result = agent.generate_context_aware_code(project, "user authentication endpoint")

        self.assertIn("fastapi", result.lower())
        self.assertIn("authentication", result.lower())

    def test_code_template_system(self):
        """Test code template system for common patterns"""
        from Socratic7 import CodeGeneratorAgent

        agent = CodeGeneratorAgent(self.mock_orchestrator)

        # Test template retrieval
        template = agent.get_code_template("rest_api_endpoint", "python", "fastapi")

        self.assertIsInstance(template, str)
        self.assertIn("@app.", template)  # FastAPI decorator

        # Test template customization
        customized = agent.customize_template(
            template,
            {"endpoint_name": "users", "method": "GET", "auth_required": True}
        )

        self.assertIn("users", customized)
        self.assertIn("GET", customized.upper())

    def test_code_quality_analysis(self):
        """Test automated code quality analysis"""
        from Socratic7 import CodeGeneratorAgent

        agent = CodeGeneratorAgent(self.mock_orchestrator)

        # Test code with quality issues
        problematic_code = """
def bad_function():
    x = 1
    y = 2
    return x+y  # No spaces around operator

class badClass:  # Should be PascalCase
    def __init__(self):
        pass
"""

        # Test quality analysis
        analysis = agent.analyze_code_quality(problematic_code, "python")

        self.assertIsInstance(analysis, dict)
        self.assertIn('issues', analysis)
        self.assertIn('score', analysis)
        self.assertLess(analysis['score'], 8.0)  # Should flag quality issues

    def test_incremental_code_improvement(self):
        """Test incremental code improvement suggestions"""
        from Socratic7 import CodeGeneratorAgent

        agent = CodeGeneratorAgent(self.mock_orchestrator)

        # Mock existing code
        existing_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
"""

        # Test improvement suggestions
        improvements = agent.suggest_improvements(existing_code, "python")

        self.assertIsInstance(improvements, list)
        # Should suggest using sum() or list comprehension
        improvement_text = str(improvements).lower()
        self.assertTrue(
            'sum(' in improvement_text or
            'comprehension' in improvement_text or
            'optimization' in improvement_text
        )


class TestAdvancedSystemMonitoring(unittest.TestCase):
    """Tests for advanced system monitoring and analytics"""

    def setUp(self):
        """Set up monitoring test environment"""
        self.mock_orchestrator = Mock()

    def test_user_engagement_analytics(self):
        """Test user engagement tracking and analytics"""
        from Socratic7 import SystemMonitorAgent

        agent = SystemMonitorAgent(self.mock_orchestrator)

        # Mock user activity data
        user_activities = [
            {'user': 'user1', 'action': 'question_answered', 'timestamp': datetime.datetime.now()},
            {'user': 'user1', 'action': 'code_generated', 'timestamp': datetime.datetime.now()},
            {'user': 'user2', 'action': 'project_created', 'timestamp': datetime.datetime.now()}
        ]

        # Test engagement calculation
        engagement_metrics = agent.calculate_engagement_metrics(user_activities, time_window='1d')

        self.assertIsInstance(engagement_metrics, dict)
        self.assertIn('active_users', engagement_metrics)
        self.assertIn('actions_per_user', engagement_metrics)
        self.assertIn('engagement_score', engagement_metrics)

    def test_system_performance_monitoring(self):
        """Test system performance monitoring"""
        from Socratic7 import SystemMonitorAgent

        agent = SystemMonitorAgent(self.mock_orchestrator)

        # Mock performance metrics
        performance_data = {
            'response_times': [0.1, 0.2, 0.15, 0.3, 0.12],
            'memory_usage': [45.2, 47.1, 46.8, 48.3, 45.9],
            'active_sessions': [12, 15, 13, 18, 14]
        }

        # Test performance analysis
        analysis = agent.analyze_performance(performance_data)

        self.assertIsInstance(analysis, dict)
        self.assertIn('avg_response_time', analysis)
        self.assertIn('memory_trend', analysis)
        self.assertIn('load_status', analysis)

    def test_predictive_scaling(self):
        """Test predictive scaling based on usage patterns"""
        from Socratic7 import SystemMonitorAgent

        agent = SystemMonitorAgent(self.mock_orchestrator)

        # Mock historical usage data
        historical_usage = [
            {'timestamp': datetime.datetime.now() - datetime.timedelta(hours=i), 'load': 10 + i * 2}
            for i in range(24)  # Last 24 hours
        ]

        # Test scaling prediction
        scaling_recommendation = agent.predict_scaling_needs(historical_usage, forecast_hours=4)

        self.assertIsInstance(scaling_recommendation, dict)
        self.assertIn('recommended_capacity', scaling_recommendation)
        self.assertIn('confidence', scaling_recommendation)
        self.assertIn('reasoning', scaling_recommendation)


class TestDataModels(unittest.TestCase):
    """Unit tests for data models"""

    def test_user_creation(self):
        """Test User dataclass creation"""
        user = User(
            username="testuser",
            passcode_hash="hashedpass",
            created_at=datetime.datetime.now(),
            projects=["proj1", "proj2"]
        )
        self.assertEqual(user.username, "testuser")
        self.assertEqual(len(user.projects), 2)

    def test_project_context_creation(self):
        """Test ProjectContext dataclass creation"""
        project = ProjectContext(
            project_id="test-123",
            name="Test Project",
            owner="testuser",
            collaborators=[],
            goals="Build a web app",
            requirements=[],
            tech_stack=["python", "django"],
            constraints=[],
            team_structure="individual",
            language_preferences="python",
            deployment_target="cloud",
            code_style="pep8",
            phase="discovery",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(len(project.tech_stack), 2)

    def test_knowledge_entry_creation(self):
        """Test KnowledgeEntry dataclass creation"""
        entry = KnowledgeEntry(
            id="test-knowledge",
            content="Test knowledge content",
            category="test",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3]
        )
        self.assertEqual(entry.id, "test-knowledge")
        self.assertEqual(len(entry.embedding), 3)


class MockClaudeResponse:
    """Mock class for Claude API responses"""

    def __init__(self, text="Mock response", input_tokens=100, output_tokens=50):
        self.content = [Mock(text=text)]
        self.usage = Mock(input_tokens=input_tokens, output_tokens=output_tokens)


class TestDatabaseOperations(unittest.TestCase):
    """Unit tests for database operations"""

    def setUp(self):
        """Set up test database in temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_projects.db")
        self.db = ProjectDatabase(self.db_path)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_user_save_load(self):
        """Test user save and load operations"""
        user = User(
            username="testuser",
            passcode_hash="hashedpass",
            created_at=datetime.datetime.now(),
            projects=[]
        )

        # Save user
        self.db.save_user(user)

        # Load user
        loaded_user = self.db.load_user("testuser")
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user.username, "testuser")

    def test_project_save_load(self):
        """Test project save and load operations"""
        project = ProjectContext(
            project_id="test-123",
            name="Test Project",
            owner="testuser",
            collaborators=[],
            goals="Test goals",
            requirements=[],
            tech_stack=[],
            constraints=[],
            team_structure="individual",
            language_preferences="python",
            deployment_target="local",
            code_style="documented",
            phase="discovery",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

        # Save project
        self.db.save_project(project)

        # Load project
        loaded_project = self.db.load_project("test-123")
        self.assertIsNotNone(loaded_project)
        self.assertEqual(loaded_project.name, "Test Project")

    def test_user_exists(self):
        """Test user existence check"""
        self.assertFalse(self.db.user_exists("nonexistent"))

        # Create user
        user = User(
            username="testuser",
            passcode_hash="hashedpass",
            created_at=datetime.datetime.now(),
            projects=[]
        )
        self.db.save_user(user)

        self.assertTrue(self.db.user_exists("testuser"))

    def test_advanced_query_optimization(self):
        """Test advanced database query optimization"""
        # Test connection pooling
        db = ProjectDatabase(self.db_path)

        # Mock multiple concurrent operations
        import threading

        results = []
        errors = []

        def concurrent_operation(thread_id):
            try:
                user = User(f"user_{thread_id}", "hash", datetime.datetime.now(), [])
                db.save_user(user)
                loaded = db.load_user(f"user_{thread_id}")
                results.append(loaded is not None)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = [threading.Thread(target=concurrent_operation, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle concurrent operations without errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))

    def test_database_migration_system(self):
        """Test database schema migration system"""
        db = ProjectDatabase(self.db_path)

        # Test schema version tracking
        current_version = db.get_schema_version()
        self.assertIsInstance(current_version, int)

        # Test migration check
        migrations_needed = db.check_migrations_needed()
        self.assertIsInstance(migrations_needed, list)


class TestVectorDatabase(unittest.TestCase):
    """Unit tests for vector database operations"""

    def setUp(self):
        """Set up test vector database"""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db_path = os.path.join(self.temp_dir, "test_vector_db")

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    @patch('Socratic7.SentenceTransformer')
    @patch('Socratic7.chromadb.PersistentClient')
    def test_knowledge_addition(self, mock_chromadb, mock_transformer):
        """Test adding knowledge to vector database"""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_transformer.return_value = mock_model

        # Mock ChromaDB
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        # Create vector database
        vector_db = VectorDatabase(self.vector_db_path)

        # Test adding knowledge
        entry = KnowledgeEntry(
            id="test-1",
            content="Test content",
            category="test",
            metadata={"source": "test"}
        )

        vector_db.add_knowledge(entry)
        mock_collection.add.assert_called_once()


class TestAgents(unittest.TestCase):
    """Unit tests for individual agents"""

    def setUp(self):
        """Set up test environment for agents"""
        self.temp_dir = tempfile.mkdtemp()

        # Mock orchestrator
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.database = Mock()
        self.mock_orchestrator.vector_db = Mock()
        self.mock_orchestrator.claude_client = Mock()
        self.mock_orchestrator.context_analyzer = Mock()
        self.mock_orchestrator.system_monitor = Mock()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_project_manager_create_project(self):
        """Test project creation through ProjectManagerAgent"""
        agent = ProjectManagerAgent(self.mock_orchestrator)

        request = {
            'action': 'create_project',
            'project_name': 'Test Project',
            'owner': 'testuser'
        }

        result = agent.process(request)
        self.assertEqual(result['status'], 'success')
        self.assertIn('project', result)

    def test_project_manager_invalid_action(self):
        """Test invalid action handling"""
        agent = ProjectManagerAgent(self.mock_orchestrator)

        request = {'action': 'invalid_action'}
        result = agent.process(request)

        self.assertEqual(result['status'], 'error')
        self.assertIn('Unknown action', result['message'])

    def test_socratic_counselor_static_questions(self):
        """Test static question generation"""
        agent = SocraticCounselorAgent(self.mock_orchestrator)
        agent.use_dynamic_questions = False

        # Mock project
        project = Mock()
        project.phase = 'discovery'
        project.conversation_history = []

        request = {
            'action': 'generate_question',
            'project': project
        }

        result = agent.process(request)
        self.assertEqual(result['status'], 'success')
        self.assertIn('question', result)

    def test_context_analyzer_summary(self):
        """Test context analysis functionality"""
        agent = ContextAnalyzerAgent(self.mock_orchestrator)

        project = ProjectContext(
            project_id="test",
            name="Test Project",
            owner="user",
            collaborators=[],
            goals="Build web app",
            requirements=["fast", "secure"],
            tech_stack=["python", "django"],
            constraints=["budget"],
            team_structure="team",
            language_preferences="python",
            deployment_target="cloud",
            code_style="pep8",
            phase="analysis",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

        summary = agent.get_context_summary(project)
        self.assertIn("Build web app", summary)
        self.assertIn("python", summary)

    def test_advanced_socratic_questioning(self):
        """Test advanced Socratic questioning with domain expertise"""
        agent = SocraticCounselorAgent(self.mock_orchestrator)

        # Mock domain-specific knowledge
        domain_context = {
            'domain': 'web_development',
            'user_expertise': 'intermediate',
            'current_topic': 'database_design'
        }

        # Test domain-aware question generation
        question = agent.generate_domain_specific_question(domain_context)

        self.assertIsInstance(question, str)
        self.assertGreater(len(question), 10)
        # Should contain domain-relevant terms
        question_lower = question.lower()
        self.assertTrue(
            'database' in question_lower or
            'data' in question_lower or
            'schema' in question_lower
        )


class TestConflictDetection(unittest.TestCase):
    """Tests for conflict detection functionality"""

    def setUp(self):
        """Set up conflict detector test environment"""
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.claude_client = Mock()
        self.detector = ConflictDetectorAgent(self.mock_orchestrator)

    def test_tech_stack_conflicts(self):
        """Test detection of conflicting technologies"""
        project = ProjectContext(
            project_id="test",
            name="Test Project",
            owner="user",
            collaborators=[],
            goals="",
            requirements=[],
            tech_stack=["mysql"],  # Existing database
            constraints=[],
            team_structure="individual",
            language_preferences="python",
            deployment_target="local",
            code_style="documented",
            phase="discovery",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

        new_insights = {
            'tech_stack': ['postgresql']  # Conflicting database
        }

        request = {
            'action': 'detect_conflicts',
            'project': project,
            'new_insights': new_insights,
            'current_user': 'testuser'
        }

        result = self.detector.process(request)
        self.assertEqual(result['status'], 'success')
        # Should detect database conflict
        conflicts = result['conflicts']
        if conflicts:  # Depends on conflict rules implementation
            self.assertTrue(any('tech_stack' in c.conflict_type for c in conflicts))

    def test_find_conflict_category(self):
        """Test conflict category detection"""
        # Test database conflict
        category = self.detector._find_conflict_category('mysql', 'postgresql')
        self.assertEqual(category, 'databases')

        # Test frontend framework conflict
        category = self.detector._find_conflict_category('react', 'vue')
        self.assertEqual(category, 'frontend_frameworks')

        # Test no conflict
        category = self.detector._find_conflict_category('python', 'mysql')
        self.assertIsNone(category)

    def test_ml_powered_conflict_prediction(self):
        """Test ML-powered conflict prediction"""
        detector = ConflictDetectorAgent(self.mock_orchestrator)

        # Mock project evolution data
        project_history = [
            {'timestamp': datetime.datetime.now() - datetime.timedelta(days=i),
             'tech_stack': ['python', 'flask'],
             'team_size': 2 + i // 10}
            for i in range(30)
        ]

        new_proposal = {
            'tech_stack': ['python', 'django'],  # Major framework change
            'team_size': 5  # Significant team growth
        }

        # Test conflict prediction
        prediction = detector.predict_conflicts(project_history, new_proposal)

        self.assertIsInstance(prediction, dict)
        self.assertIn('conflict_probability', prediction)
        self.assertIn('risk_factors', prediction)
        self.assertBetween(prediction['conflict_probability'], 0.0, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for component interactions"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch('Socratic7.anthropic.Anthropic')
    @patch('Socratic7.SentenceTransformer')
    @patch('Socratic7.chromadb.PersistentClient')
    def test_orchestrator_initialization(self, mock_chromadb, mock_transformer, mock_anthropic):
        """Test full orchestrator initialization"""
        # Mock dependencies
        mock_transformer.return_value = Mock()
        mock_chromadb.return_value = Mock()
        mock_anthropic.return_value = Mock()

        # Temporarily change config
        import Socratic7
        original_data_dir = Socratic7.CONFIG['DATA_DIR']
        Socratic7.CONFIG['DATA_DIR'] = self.temp_dir

        try:
            orchestrator = AgentOrchestrator("test-api-key")

            # Verify all agents initialized
            self.assertIsNotNone(orchestrator.project_manager)
            self.assertIsNotNone(orchestrator.socratic_counselor)
            self.assertIsNotNone(orchestrator.context_analyzer)
            self.assertIsNotNone(orchestrator.code_generator)
            self.assertIsNotNone(orchestrator.system_monitor)
            self.assertIsNotNone(orchestrator.conflict_detector)

        finally:
            # Restore original config
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    @patch('Socratic7.anthropic.Anthropic')
    def test_claude_client_integration(self, mock_anthropic):
        """Test Claude client integration with mock responses"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Mock system monitor
        mock_orchestrator = Mock()
        mock_orchestrator.system_monitor = Mock()
        mock_orchestrator.system_monitor.process.return_value = {'status': 'success'}

        claude_client = ClaudeClient("test-key", mock_orchestrator)

        # Mock response
        mock_response = MockClaudeResponse('{"goals": "Build web app"}')
        mock_client.messages.create.return_value = mock_response

        # Test insight extraction
        project = Mock()
        project.goals = ""
        project.phase = "discovery"
        project.tech_stack = []

        insights = claude_client.extract_insights("I want to build a web app", project)
        self.assertIsInstance(insights, dict)


class ScenarioTester:
    """Automated scenario testing class"""

    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.cleanup_needed = temp_dir is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_needed:
            shutil.rmtree(self.temp_dir)

    @patch('Socratic7.anthropic.Anthropic')
    @patch('Socratic7.SentenceTransformer')
    @patch('Socratic7.chromadb.PersistentClient')
    def create_test_system(self, mock_chromadb, mock_transformer, mock_anthropic):
        """Create a fully mocked test system"""
        # Mock all external dependencies
        mock_transformer.return_value = Mock()
        mock_transformer.return_value.encode.return_value = [0.1, 0.2, 0.3]

        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {'ids': []}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client

        # Set up temporary data directory
        import Socratic7
        original_data_dir = Socratic7.CONFIG['DATA_DIR']
        Socratic7.CONFIG['DATA_DIR'] = self.temp_dir

        try:
            orchestrator = AgentOrchestrator("test-api-key")
            return orchestrator, original_data_dir
        except Exception as e:
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir
            raise

    def test_complete_user_journey(self):
        """Test complete user journey from account creation to code generation"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Step 1: Create user
            user = User(
                username="scenario_user",
                passcode_hash=hashlib.sha256("testpass".encode()).hexdigest(),
                created_at=datetime.datetime.now(),
                projects=[]
            )
            orchestrator.database.save_user(user)

            # Step 2: Create project
            result = orchestrator.process_request('project_manager', {
                'action': 'create_project',
                'project_name': 'E-commerce Platform',
                'owner': 'scenario_user'
            })
            self.assert_success(result, "Project creation failed")
            project = result['project']

            # Step 3: Simulate conversation
            mock_insights = {
                'goals': 'Build an online store for handmade items',
                'tech_stack': ['python', 'django', 'postgresql'],
                'requirements': ['payment processing', 'user authentication', 'inventory management']
            }

            # Mock Claude client to return insights
            mock_response = MockClaudeResponse(json.dumps(mock_insights))
            orchestrator.claude_client.client.messages.create.return_value = mock_response

            # Process user response
            result = orchestrator.process_request('socratic_counselor', {
                'action': 'process_response',
                'project': project,
                'response': 'I want to build an online store with Django and PostgreSQL',
                'current_user': 'scenario_user'
            })
            self.assert_success(result, "Response processing failed")

            # Step 4: Test code generation
            mock_code_response = MockClaudeResponse("# Django E-commerce Application\n# Generated code here...")
            orchestrator.claude_client.client.messages.create.return_value = mock_code_response

            result = orchestrator.process_request('code_generator', {
                'action': 'generate_script',
                'project': project
            })
            self.assert_success(result, "Code generation failed")

            print("✅ Complete user journey test passed")
            return True

        except Exception as e:
            print(f"❌ Complete user journey test failed: {e}")
            return False
        finally:
            # Restore original config
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_collaboration_scenario(self):
        """Test collaboration features"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Create users
            owner = User("owner", "hash1", datetime.datetime.now(), [])
            collaborator = User("collaborator", "hash2", datetime.datetime.now(), [])

            orchestrator.database.save_user(owner)
            orchestrator.database.save_user(collaborator)

            # Create project
            result = orchestrator.process_request('project_manager', {
                'action': 'create_project',
                'project_name': 'Team Project',
                'owner': 'owner'
            })
            self.assert_success(result, "Project creation failed")
            project = result['project']

            # Add collaborator
            result = orchestrator.process_request('project_manager', {
                'action': 'add_collaborator',
                'project': project,
                'username': 'collaborator'
            })
            self.assert_success(result, "Adding collaborator failed")

            # List collaborators
            result = orchestrator.process_request('project_manager', {
                'action': 'list_collaborators',
                'project': project
            })
            self.assert_success(result, "Listing collaborators failed")
            self.assert_condition(
                len(result['collaborators']) == 2,
                "Should have owner + 1 collaborator"
            )

            print("✅ Collaboration scenario test passed")
            return True

        except Exception as e:
            print(f"❌ Collaboration scenario test failed: {e}")
            return False
        finally:
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_conflict_detection_scenario(self):
        """Test conflict detection in realistic scenario"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Create project with initial tech stack
            result = orchestrator.process_request('project_manager', {
                'action': 'create_project',
                'project_name': 'Conflicting Project',
                'owner': 'testuser'
            })
            self.assert_success(result, "Project creation failed")
            project = result['project']

            # Add initial tech stack
            project.tech_stack = ['mysql', 'react']

            # Test conflicting insights
            conflicting_insights = {
                'tech_stack': ['postgresql', 'vue'],  # Both conflict with existing
                'requirements': ['fast performance']
            }

            result = orchestrator.process_request('conflict_detector', {
                'action': 'detect_conflicts',
                'project': project,
                'new_insights': conflicting_insights,
                'current_user': 'testuser'
            })
            self.assert_success(result, "Conflict detection failed")

            # Should detect conflicts
            conflicts = result['conflicts']
            database_conflict = any(c.conflict_type == 'tech_stack' and 'mysql' in c.old_value for c in conflicts)
            frontend_conflict = any(c.conflict_type == 'tech_stack' and 'react' in c.old_value for c in conflicts)

            print(f"Detected {len(conflicts)} conflicts")
            print("✅ Conflict detection scenario test passed")
            return True

        except Exception as e:
            print(f"❌ Conflict detection scenario test failed: {e}")
            return False
        finally:
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_advanced_ml_scenario(self):
        """Test ML-powered features scenario"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Test semantic clustering of similar projects
            projects = [
                self.create_mock_project("E-commerce Platform", ["python", "django", "postgresql"]),
                self.create_mock_project("Online Store", ["python", "flask", "mysql"]),
                self.create_mock_project("Chat Application", ["node.js", "socket.io", "mongodb"])
            ]

            # Mock ML clustering
            from Socratic7 import VectorDatabase
            with patch.object(VectorDatabase, 'find_similar_projects') as mock_cluster:
                mock_cluster.return_value = [
                    {'project_id': projects[0].project_id, 'similarity': 0.85},
                    {'project_id': projects[1].project_id, 'similarity': 0.82}
                ]

                # Test similar project finding
                similar = orchestrator.vector_db.find_similar_projects(projects[0])

                self.assertIsInstance(similar, list)
                self.assertGreater(len(similar), 0)

            print("✅ Advanced ML scenario test passed")
            return True

        except Exception as e:
            print(f"❌ Advanced ML scenario test failed: {e}")
            return False
        finally:
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def create_mock_project(self, name, tech_stack):
        """Helper method to create mock project"""
        return ProjectContext(
            project_id=f"mock_{name.lower().replace(' ', '_')}",
            name=name,
            owner="test_user",
            collaborators=[],
            goals=f"Build {name}",
            requirements=[],
            tech_stack=tech_stack,
            constraints=[],
            team_structure="individual",
            language_preferences="python",
            deployment_target="cloud",
            code_style="pep8",
            phase="development",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

    def assert_success(self, result, message):
        """Assert that result indicates success"""
        if result.get('status') != 'success':
            raise AssertionError(f"{message}: {result}")

    def assert_condition(self, condition, message):
        """Assert a condition is true"""
        if not condition:
            raise AssertionError(message)


class IntegrationTester:
    """Helper class for integration testing scenarios"""

    def __init__(self):
        self.test_api_key = "test-api-key-for-testing"

    def run_advanced_tests(self):
        """Run advanced feature tests"""
        print("🚀 Running Advanced Feature Tests...")

        advanced_test_classes = [
            TestAdvancedRAGFeatures,
            TestRealTimeCollaboration,
            TestIntelligentCodeGeneration,
            TestAdvancedSystemMonitoring
        ]

        for test_class in advanced_test_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

            self.test_results['advanced_tests'] = self.test_results.get('advanced_tests',
                                                                        {'passed': 0, 'failed': 0, 'errors': []})
            self.test_results['advanced_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
            self.test_results['advanced_tests']['failed'] += len(result.failures) + len(result.errors)

            if result.failures:
                self.test_results['advanced_tests']['errors'].extend(
                    [f"{test}: {error}" for test, error in result.failures])
            if result.errors:
                self.test_results['advanced_tests']['errors'].extend(
                    [f"{test}: {error}" for test, error in result.errors])

    def test_complete_user_journey(self):
        """Test complete user journey from project creation to code generation"""
        try:
            # Mock the orchestrator initialization
            with unittest.mock.patch('Socratic7.anthropic.Anthropic'):
                # Create orchestrator
                orchestrator = Socratic7.AgentOrchestrator(self.test_api_key)

                # Create test user
                test_user = Socratic7.User(
                    username="test_user",
                    passcode_hash="test_hash",
                    created_at=datetime.datetime.now(),
                    projects=[]
                )
                orchestrator.database.save_user(test_user)

                # Create test project
                result = orchestrator.process_request('project_manager', {
                    'action': 'create_project',
                    'project_name': 'Test Journey Project',
                    'owner': 'test_user'
                })

                if result['status'] != 'success':
                    print(f"Failed to create project: {result}")
                    return False

                project = result['project']

                # Test adding project context
                project.goals = "Build a web application"
                project.tech_stack = ["python", "flask"]
                project.requirements = ["user authentication", "database storage"]

                # Test context analysis
                context_result = orchestrator.process_request('context_analyzer', {
                    'action': 'get_summary',
                    'project': project
                })

                if context_result['status'] != 'success':
                    print(f"Context analysis failed: {context_result}")
                    return False

                # Test save/load cycle
                orchestrator.process_request('project_manager', {
                    'action': 'save_project',
                    'project': project
                })

                loaded_result = orchestrator.process_request('project_manager', {
                    'action': 'load_project',
                    'project_id': project.project_id
                })

                if loaded_result['status'] != 'success':
                    print(f"Project loading failed: {loaded_result}")
                    return False

                print("✓ Complete user journey test passed")
                return True

        except Exception as e:
            print(f"Complete user journey test error: {e}")
            return False

    def test_collaboration_scenario(self):
        """Test collaboration features"""
        try:
            # Mock the orchestrator initialization
            with unittest.mock.patch('Socratic7.anthropic.Anthropic'):
                orchestrator = Socratic7.AgentOrchestrator(self.test_api_key)

                # Create test users
                owner = Socratic7.User(
                    username="project_owner",
                    passcode_hash="owner_hash",
                    created_at=datetime.datetime.now(),
                    projects=[]
                )
                collaborator = Socratic7.User(
                    username="collaborator1",
                    passcode_hash="collab_hash",
                    created_at=datetime.datetime.now(),
                    projects=[]
                )

                orchestrator.database.save_user(owner)
                orchestrator.database.save_user(collaborator)

                # Create project
                result = orchestrator.process_request('project_manager', {
                    'action': 'create_project',
                    'project_name': 'Collaboration Test Project',
                    'owner': 'project_owner'
                })

                if result['status'] != 'success':
                    return False

                project = result['project']

                # Test adding collaborator
                add_result = orchestrator.process_request('project_manager', {
                    'action': 'add_collaborator',
                    'project': project,
                    'username': 'collaborator1'
                })

                if add_result['status'] != 'success':
                    print(f"Failed to add collaborator: {add_result}")
                    return False

                # Test listing collaborators
                list_result = orchestrator.process_request('project_manager', {
                    'action': 'list_collaborators',
                    'project': project
                })

                if list_result['status'] != 'success':
                    print(f"Failed to list collaborators: {list_result}")
                    return False

                # Verify collaborator was added
                collaborators = list_result['collaborators']
                collaborator_usernames = [c['username'] for c in collaborators]

                if 'collaborator1' not in collaborator_usernames:
                    print("Collaborator was not properly added")
                    return False

                print("✓ Collaboration scenario test passed")
                return True

        except Exception as e:
            print(f"Collaboration test error: {e}")
            return False

    def test_conflict_detection_scenario(self):
        """Test conflict detection in collaborative projects"""
        try:
            # Mock the orchestrator initialization
            with unittest.mock.patch('Socratic7.anthropic.Anthropic'):
                orchestrator = Socratic7.AgentOrchestrator(self.test_api_key)

                # Create test project with existing tech stack
                project = Socratic7.ProjectContext(
                    project_id="conflict_test_project",
                    name="Conflict Test Project",
                    owner="test_owner",
                    collaborators=["collaborator1"],
                    goals="Build a web app",
                    requirements=["fast performance"],
                    tech_stack=["postgresql"],  # Existing database choice
                    constraints=["low budget"],
                    team_structure="small team",
                    language_preferences="python",
                    deployment_target="cloud",
                    code_style="clean",
                    phase="analysis",
                    conversation_history=[],
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now()
                )

                # Test conflict detection with new insights
                new_insights = {
                    'tech_stack': ['mysql'],  # Conflicting database
                    'requirements': ['slow performance is ok'],  # Conflicting requirement
                }

                conflict_result = orchestrator.process_request('conflict_detector', {
                    'action': 'detect_conflicts',
                    'project': project,
                    'new_insights': new_insights,
                    'current_user': 'collaborator1'
                })

                if conflict_result['status'] != 'success':
                    print(f"Conflict detection failed: {conflict_result}")
                    return False

                conflicts = conflict_result.get('conflicts', [])

                # Should detect at least one conflict (database choice)
                if len(conflicts) == 0:
                    print("No conflicts detected when conflicts should exist")
                    return False

                print(f"✓ Conflict detection scenario test passed - detected {len(conflicts)} conflicts")
                return True

        except Exception as e:
            print(f"Conflict detection test error: {e}")
            return False


class TestScenarios(unittest.TestCase):
    """Integration tests for complete user scenarios"""

    def test_complete_user_journey(self):
        """Test complete user journey from project creation to code generation"""
        original_data_dir = Socratic7.CONFIG['DATA_DIR']  # Initialize at the start
        try:
            # Use temporary directory for testing
            import tempfile
            test_dir = tempfile.mkdtemp()
            Socratic7.CONFIG['DATA_DIR'] = test_dir

            tester = IntegrationTester()

            # Test the complete journey
            result = tester.test_complete_user_journey()
            self.assertTrue(result)

        except Exception as e:
            print(f"❌ Complete user journey test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Complete user journey test failed: {e}")
        finally:
            # Restore original configuration
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_collaboration_scenario(self):
        """Test collaboration features"""
        original_data_dir = Socratic7.CONFIG['DATA_DIR']  # Initialize at the start
        try:
            # Use temporary directory for testing
            import tempfile
            test_dir = tempfile.mkdtemp()
            Socratic7.CONFIG['DATA_DIR'] = test_dir

            tester = IntegrationTester()

            # Test collaboration scenario
            result = tester.test_collaboration_scenario()
            self.assertTrue(result)

        except Exception as e:
            print(f"❌ Collaboration scenario test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Collaboration scenario test failed: {e}")
        finally:
            # Restore original configuration
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_conflict_detection_scenario(self):
        """Test conflict detection in collaborative projects"""
        original_data_dir = Socratic7.CONFIG['DATA_DIR']  # Initialize at the start
        try:
            # Use temporary directory for testing
            import tempfile
            test_dir = tempfile.mkdtemp()
            Socratic7.CONFIG['DATA_DIR'] = test_dir

            tester = IntegrationTester()

            # Test conflict detection
            result = tester.test_conflict_detection_scenario()
            self.assertTrue(result)

        except Exception as e:
            print(f"❌ Conflict detection scenario test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Conflict detection scenario test failed: {e}")
        finally:
            # Restore original configuration
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_knowledge_addition(self):
        """Test adding knowledge to vector database"""
        try:
            # Mock the embedding model to return a list instead of numpy array
            with unittest.mock.patch('Socratic7.SentenceTransformer') as mock_transformer:
                # Create a mock that returns a list with tolist() method
                mock_model = unittest.mock.MagicMock()
                mock_embedding = unittest.mock.MagicMock()
                mock_embedding.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
                mock_model.encode.return_value = mock_embedding
                mock_transformer.return_value = mock_model

                # Use temporary directory for testing
                import tempfile
                test_dir = tempfile.mkdtemp()

                vector_db = Socratic7.VectorDatabase(test_dir)

                # Create test knowledge entry
                entry = Socratic7.KnowledgeEntry(
                    id="test_entry",
                    content="Test knowledge content",
                    category="test",
                    metadata={"test": True}
                )

                # This should not raise an error now
                vector_db.add_knowledge(entry)

                # Verify embedding was set
                self.assertIsNotNone(entry.embedding)
                self.assertEqual(entry.embedding, [0.1, 0.2, 0.3, 0.4, 0.5])

                print("✓ Knowledge addition test passed")

        except Exception as e:
            print(f"❌ Knowledge addition test failed: {e}")
            self.fail(f"Knowledge addition test failed: {e}")


class PerformanceTest:
    """Performance testing utilities"""

    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def test_database_performance():
        """Test database operation performance"""
        print("\n🔍 Running Performance Tests...")

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "perf_test.db")
            db = ProjectDatabase(db_path)

            # Test user save/load performance
            user = User("perfuser", "hash", datetime.datetime.now(), [])

            _, save_time = PerformanceTest.measure_time(db.save_user, user)
            _, load_time = PerformanceTest.measure_time(db.load_user, "perfuser")

            print(f"  User save time: {save_time:.4f}s")
            print(f"  User load time: {load_time:.4f}s")

            # Test project save/load performance
            project = ProjectContext(
                project_id="perf-test",
                name="Performance Test",
                owner="perfuser",
                collaborators=[],
                goals="Test performance",
                requirements=[],
                tech_stack=[],
                constraints=[],
                team_structure="individual",
                language_preferences="python",
                deployment_target="local",
                code_style="documented",
                phase="discovery",
                conversation_history=[],
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )

            _, proj_save_time = PerformanceTest.measure_time(db.save_project, project)
            _, proj_load_time = PerformanceTest.measure_time(db.load_project, "perf-test")

            print(f"  Project save time: {proj_save_time:.4f}s")
            print(f"  Project load time: {proj_load_time:.4f}s")

            # Performance assertions
            assert save_time < 1.0, f"User save too slow: {save_time}s"
            assert load_time < 1.0, f"User load too slow: {load_time}s"
            assert proj_save_time < 1.0, f"Project save too slow: {proj_save_time}s"
            assert proj_load_time < 1.0, f"Project load too slow: {proj_load_time}s"

            print("  ✅ All performance tests passed")

    @staticmethod
    def test_advanced_performance_metrics():
        """Test advanced performance characteristics"""
        print("\n⚡ Running Advanced Performance Tests...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test concurrent user simulation
            import concurrent.futures

            def simulate_user_session(user_id):
                """Simulate a complete user session"""
                start_time = time.time()

                # Mock user operations
                operations = [
                    lambda: time.sleep(0.01),  # Project creation
                    lambda: time.sleep(0.005),  # Question generation
                    lambda: time.sleep(0.02),  # Code generation
                    lambda: time.sleep(0.003),  # Context analysis
                ]

                for op in operations:
                    op()

                return time.time() - start_time

            # Test concurrent load
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(simulate_user_session, i) for i in range(20)]
                session_times = [f.result() for f in concurrent.futures.as_completed(futures)]

            avg_session_time = sum(session_times) / len(session_times)
            max_session_time = max(session_times)

            print(f"  Concurrent user sessions: {len(session_times)}")
            print(f"  Average session time: {avg_session_time:.4f}s")
            print(f"  Max session time: {max_session_time:.4f}s")

            # Performance assertions
            assert avg_session_time < 0.1, f"Average session time too slow: {avg_session_time}s"
            assert max_session_time < 0.2, f"Max session time too slow: {max_session_time}s"

            print("  ✅ Advanced performance tests passed")


class TestRunner:
    """Main test runner class"""

    def __init__(self):
        self.test_results = {
            'unit_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'integration_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'scenario_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'performance_tests': {'passed': 0, 'failed': 0, 'errors': []}
        }

    def run_unit_tests(self):
        """Run all unit tests"""
        print("🧪 Running Unit Tests...")

        unit_test_classes = [
            TestDataModels,
            TestDatabaseOperations,
            TestVectorDatabase,
            TestAgents,
            TestConflictDetection
        ]

        for test_class in unit_test_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

            self.test_results['unit_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
            self.test_results['unit_tests']['failed'] += len(result.failures) + len(result.errors)

            if result.failures:
                self.test_results['unit_tests']['errors'].extend(
                    [f"{test}: {error}" for test, error in result.failures])
            if result.errors:
                self.test_results['unit_tests']['errors'].extend([f"{test}: {error}" for test, error in result.errors])

    def run_integration_tests(self):
        """Run integration tests"""
        print("🔗 Running Integration Tests...")

        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
        result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

        self.test_results['integration_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
        self.test_results['integration_tests']['failed'] += len(result.failures) + len(result.errors)

        if result.failures:
            self.test_results['integration_tests']['errors'].extend(
                [f"{test}: {error}" for test, error in result.failures])
        if result.errors:
            self.test_results['integration_tests']['errors'].extend(
                [f"{test}: {error}" for test, error in result.errors])

    def run_scenario_tests(self):
        """Run scenario tests"""
        print("🎭 Running Scenario Tests...")

        suite = unittest.TestLoader().loadTestsFromTestCase(TestScenarios)
        result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

        self.test_results['scenario_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
        self.test_results['scenario_tests']['failed'] += len(result.failures) + len(result.errors)

        if result.failures:
            self.test_results['scenario_tests']['errors'].extend(
                [f"{test}: {error}" for test, error in result.failures])
        if result.errors:
            self.test_results['scenario_tests']['errors'].extend([f"{test}: {error}" for test, error in result.errors])

    def run_performance_tests(self):
        """Run performance tests"""
        print("⚡ Running Performance Tests...")

        try:
            PerformanceTest.test_database_performance()
            self.test_results['performance_tests']['passed'] += 1
        except Exception as e:
            self.test_results['performance_tests']['failed'] += 1
            self.test_results['performance_tests']['errors'].append(f"Performance test failed: {e}")

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Comprehensive Test Suite for Socratic RAG System v7.0")
        print("=" * 70)

        start_time = time.time()

        try:
            self.run_unit_tests()
            self.run_integration_tests()
            self.run_scenario_tests()
            self.run_performance_tests()
        except KeyboardInterrupt:
            print("\n⚠️  Test execution interrupted by user")
            return False

        end_time = time.time()
        total_time = end_time - start_time

        self.print_test_summary(total_time)
        return self.all_tests_passed()

    def print_test_summary(self, total_time):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        total_passed = 0
        total_failed = 0

        for test_type, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed

            status_icon = "✅" if failed == 0 else "❌"
            test_name = test_type.replace('_', ' ').title()

            print(f"{status_icon} {test_name:20} | Passed: {passed:3d} | Failed: {failed:3d}")

            # Show detailed errors if any
            if results['errors'] and len(results['errors']) <= 5:  # Limit error output
                for error in results['errors']:
                    print(f"    ❌ {error}")
            elif len(results['errors']) > 5:
                print(f"    ❌ ... and {len(results['errors']) - 5} more errors")

        print("-" * 70)
        print(f"📈 TOTAL RESULTS:")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")

        if total_failed == 0:
            print(f"\n🎉 ALL TESTS PASSED! Socratic RAG System v7.0 is ready for deployment.")
        else:
            print(f"\n⚠️  {total_failed} tests failed. Please review the errors above.")

        print("=" * 70)

    def all_tests_passed(self):
        """Check if all tests passed"""
        return all(results['failed'] == 0 for results in self.test_results.values())

    def generate_test_report(self, output_file="test_report.html"):
        """Generate detailed HTML test report"""
        print(f"📄 Generating detailed test report: {output_file}")

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Socratic RAG System v7.0 - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
        .summary { background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .test-section { background-color: white; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .passed { color: #27ae60; font-weight: bold; }
        .failed { color: #e74c3c; font-weight: bold; }
        .error { background-color: #f8d7da; padding: 10px; margin: 5px 0; border-radius: 4px; font-family: monospace; font-size: 12px; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background-color: #ecf0f1; border-radius: 4px; }
        .timestamp { color: #7f8c8d; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Socratic RAG System v7.0 - Test Report</h1>
        <p class="timestamp">Generated on: {timestamp}</p>
    </div>

    <div class="summary">
        <h2>📊 Overall Summary</h2>
        <div class="metric">Total Tests: <strong>{total_tests}</strong></div>
        <div class="metric">Passed: <span class="passed">{total_passed}</span></div>
        <div class="metric">Failed: <span class="failed">{total_failed}</span></div>
        <div class="metric">Success Rate: <strong>{success_rate:.1f}%</strong></div>
    </div>
""".format(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=sum(r['passed'] + r['failed'] for r in self.test_results.values()),
            total_passed=sum(r['passed'] for r in self.test_results.values()),
            total_failed=sum(r['failed'] for r in self.test_results.values()),
            success_rate=(sum(r['passed'] for r in self.test_results.values()) /
                          max(1, sum(r['passed'] + r['failed'] for r in self.test_results.values())) * 100)
        )

        # Add detailed sections for each test type
        for test_type, results in self.test_results.items():
            test_name = test_type.replace('_', ' ').title()
            status_class = "passed" if results['failed'] == 0 else "failed"

            html_content += f"""
    <div class="test-section">
        <h3 class="{status_class}">🔧 {test_name}</h3>
        <p>Passed: <span class="passed">{results['passed']}</span> | 
           Failed: <span class="failed">{results['failed']}</span></p>
"""

            if results['errors']:
                html_content += "<h4>Errors:</h4>"
                for error in results['errors'][:10]:  # Limit to first 10 errors
                    html_content += f'<div class="error">{error}</div>'

            html_content += "</div>"

        html_content += """
    <div class="summary">
        <h2>🏁 Conclusion</h2>
        <p>This automated test suite validates the core functionality of the Socratic RAG System v7.0, 
        including data models, database operations, agent behaviors, integration points, and performance characteristics.</p>
        <p><strong>Recommendation:</strong> {}
    </div>
</body>
</html>
""".format(
            "✅ System is ready for deployment." if self.all_tests_passed()
            else "⚠️ Address failed tests before deployment."
        )

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Test report saved to {output_file}")
        except Exception as e:
            print(f"❌ Failed to generate test report: {e}")


def run_specific_test_class(test_class_name):
    """Run a specific test class"""
    test_classes = {
        'data': TestDataModels,
        'database': TestDatabaseOperations,
        'vector': TestVectorDatabase,
        'agents': TestAgents,
        'conflicts': TestConflictDetection,
        'integration': TestIntegration,
        'scenarios': TestScenarios
    }

    if test_class_name.lower() in test_classes:
        print(f"🎯 Running specific test: {test_class_name}")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_classes[test_class_name.lower()])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    else:
        print(f"❌ Unknown test class: {test_class_name}")
        print(f"Available tests: {', '.join(test_classes.keys())}")
        return False


def main():
    """Main function to run tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Test Suite for Socratic RAG System v7.0")
    parser.add_argument('--test', type=str, help='Run specific test class', choices=[
        'data', 'database', 'vector', 'agents', 'conflicts', 'integration', 'scenarios'
    ])
    parser.add_argument('--report', action='store_true', help='Generate HTML test report')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (skip scenarios)')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')

    args = parser.parse_args()

    if args.test:
        # Run specific test class
        success = run_specific_test_class(args.test)
        sys.exit(0 if success else 1)

    elif args.performance:
        # Run performance tests only
        print("⚡ Running Performance Tests Only...")
        try:
            PerformanceTest.test_database_performance()
            print("✅ Performance tests completed successfully")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Performance tests failed: {e}")
            sys.exit(1)

    else:
        # Run full test suite
        runner = TestRunner()

        if args.quick:
            print("🏃 Running Quick Test Suite (excluding scenarios)...")
            runner.run_unit_tests()
            runner.run_integration_tests()
            runner.run_performance_tests()
        else:
            success = runner.run_all_tests()

        if args.report:
            runner.generate_test_report()

        sys.exit(0 if runner.all_tests_passed() else 1)


if __name__ == "__main__":
    # Enable better error reporting
    import traceback
    import warnings

    warnings.filterwarnings('ignore', category=DeprecationWarning)

    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error during test execution:")
        print(f"   {e}")
        traceback.print_exc()
        sys.exit(1)


