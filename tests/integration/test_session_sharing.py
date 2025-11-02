"""
Test that multiple browser connections share the same g.chat instance.

This verifies the fix for session management where all WebSocket connections
in the same container should see and update the same shared chat.
"""
from core.chat import Chat, Role
from core.globals import g


def test_session_creation_shares_chat():
    """Test that creating sessions with existing chat shares that chat instance"""
    # Create the main chat (like main.py does on startup)
    main_chat = Chat(debug_title="Main Chat")
    main_chat.set_instruction_message("System prompt")
    main_chat.add_message(Role.USER, "Hello")

    # Set as global chat
    g.chat = main_chat

    # Simulate first browser connecting
    session1 = g.create_session("session-1", chat=g.chat)

    # Simulate second browser connecting
    session2 = g.create_session("session-2", chat=g.chat)

    # Both sessions should point to the SAME chat instance
    assert session1.chat is session2.chat
    assert session1.chat is main_chat
    assert session2.chat is main_chat

    print("✓ Multiple sessions share the same chat instance")


def test_session_chat_modifications_visible_to_all():
    """Test that modifications in one session's chat are visible to all sessions"""
    # Setup
    main_chat = Chat(debug_title="Shared Chat")
    main_chat.set_instruction_message("System prompt")
    g.chat = main_chat

    # Create multiple sessions
    session1 = g.create_session("session-1", chat=g.chat)
    session2 = g.create_session("session-2", chat=g.chat)
    session3 = g.create_session("session-3", chat=g.chat)

    # All should start with same message count
    initial_count = len(main_chat.messages)
    assert len(session1.chat.messages) == initial_count
    assert len(session2.chat.messages) == initial_count
    assert len(session3.chat.messages) == initial_count

    # Add message through session1
    session1.chat.add_message(Role.USER, "Test message from browser 1")

    # All sessions should see the new message
    assert len(session1.chat.messages) == initial_count + 1
    assert len(session2.chat.messages) == initial_count + 1
    assert len(session3.chat.messages) == initial_count + 1
    assert len(main_chat.messages) == initial_count + 1

    # Verify they all have the same message
    assert session1.chat.messages[-1].content == "Test message from browser 1"
    assert session2.chat.messages[-1].content == "Test message from browser 1"
    assert session3.chat.messages[-1].content == "Test message from browser 1"

    print("✓ Chat modifications visible across all sessions")


def test_session_without_chat_creates_new():
    """Test backward compatibility: sessions without chat parameter create new chat"""
    # Clear g.chat
    g.chat = None

    # Create session without passing chat (should create new one)
    session = g.create_session("session-orphan")

    # Should have created a new chat
    assert session.chat is not None
    assert isinstance(session.chat, Chat)

    print("✓ Backward compatibility: sessions without chat create new instance")


def test_multiple_sessions_same_container():
    """Integration test: Simulate multiple browsers connecting to same container"""
    # Simulate container startup with main chat
    container_chat = Chat(debug_title="Container Main Chat")
    container_chat.set_instruction_message("You are an AI assistant")
    container_chat.add_message(Role.USER, "Initial message")
    g.chat = container_chat

    # Simulate 3 browsers connecting
    browser_sessions = []
    for i in range(1, 4):
        session = g.create_session(f"browser-{i}", chat=g.chat)
        browser_sessions.append(session)

    # All should share the same chat
    for session in browser_sessions:
        assert session.chat is container_chat

    # Add messages from different "browsers"
    # Note: Chat merges consecutive messages of same role, so these will be concatenated
    browser_sessions[0].chat.add_message(Role.USER, "Message from browser 1")
    browser_sessions[1].chat.add_message(Role.USER, "Message from browser 2")
    browser_sessions[2].chat.add_message(Role.USER, "Message from browser 3")

    # All browsers should see the merged messages (Chat concatenates same-role consecutive messages)
    # Expected: SYSTEM message + merged USER message
    for session in browser_sessions:
        assert len(session.chat.messages) == 2  # SYSTEM + merged USER
        assert session.chat.messages[0].role == Role.SYSTEM
        assert session.chat.messages[1].role == Role.USER
        # All three browser messages should be in the merged content
        assert "Initial message" in session.chat.messages[1].content
        assert "Message from browser 1" in session.chat.messages[1].content
        assert "Message from browser 2" in session.chat.messages[1].content
        assert "Message from browser 3" in session.chat.messages[1].content

    # Cleanup
    for session in browser_sessions:
        g.cleanup_session(session.session_id)

    print("✓ Multiple browsers in same container share chat correctly")
    print(f"  Simulated {len(browser_sessions)} browsers")
    print(f"  All saw {len(container_chat.messages)} messages")


if __name__ == '__main__':
    print("Testing session chat sharing...")
    print()

    test_session_creation_shares_chat()
    test_session_chat_modifications_visible_to_all()
    test_session_without_chat_creates_new()
    test_multiple_sessions_same_container()

    print()
    print("All session sharing tests passed! ✓")
