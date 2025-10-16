#!/usr/bin/env python3
"""
Cache management utility for the SearchWeb Vector Database.
Provides commands for viewing statistics, cleaning up old cache entries,
and inspecting the enhanced metadata structure.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
try:
    from dateutil import parser as date_parser
except ImportError:
    print("Error: python-dateutil is required but not installed.")
    print("Install it with: pip install python-dateutil")
    sys.exit(1)

# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.utils.search.search_vector_db import SearchResultsVectorDB


def get_db():
    """Initializes and returns the vector DB instance."""
    print("Initializing vector database...")
    db = SearchResultsVectorDB()
    if not db.is_ready:
        print("Error: Vector database is not ready. Check ChromaDB/Ollama setup.")
        sys.exit(1)
    return db


def show_stats(args):
    """Display database statistics."""
    db = get_db()
    stats = db.get_stats()
    print("\n--- SearchWeb Vector DB Stats ---")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("---------------------------------")


def prune_by_date(args):
    """Prune records older than a specified number of days."""
    db = get_db()
    cutoff_date = datetime.now() - timedelta(days=args.days)
    cutoff_iso = cutoff_date.isoformat()

    print(f"Pruning records older than {args.days} days (before {cutoff_date.strftime('%Y-%m-%d')})...")

    # Different timestamp fields for different collections
    collections = {
        "results": (db.results_collection, "retrieved_at"),
        "summaries": (db.summaries_collection, "created_at")
    }

    total_deleted = 0
    for name, (collection, timestamp_field) in collections.items():
        try:
            # ChromaDB's where filter uses '$lt' for 'less than'
            results_to_delete = collection.get(where={timestamp_field: {"$lt": cutoff_iso}})
            ids_to_delete = results_to_delete.get('ids', [])

            if ids_to_delete:
                print(f"Found {len(ids_to_delete)} old records in '{name}' collection. Deleting...")
                collection.delete(ids=ids_to_delete)
                total_deleted += len(ids_to_delete)
            else:
                print(f"No old records to delete in '{name}' collection.")
        except Exception as e:
            print(f"Error pruning '{name}' collection: {e}")

    print(f"\nPruning complete. Total records deleted: {total_deleted}")


def list_recent(args):
    """List recent cache entries."""
    db = get_db()
    try:
        # Get recent results
        print(f"\n--- Recent Search Results (last {args.limit}) ---")
        recent_results = db.results_collection.get(
            limit=args.limit,
            include=["metadatas"]
        )

        if recent_results and recent_results['metadatas']:
            for i, metadata in enumerate(recent_results['metadatas'], 1):
                retrieved_at = metadata.get('retrieved_at', '')
                # Parse ISO timestamp
                try:
                    if retrieved_at:
                        parsed_date = date_parser.parse(retrieved_at)
                        date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_str = "Unknown"
                except Exception:
                    date_str = "Invalid date"

                # Display enhanced metadata
                print(f"{i:2d}. Query: '{metadata.get('source_query', 'N/A')}' "
                      f"Type: {metadata.get('search_type', 'N/A')} "
                      f"API: {metadata.get('source_api_endpoint', 'N/A')} "
                      f"Date: {date_str}")
        else:
            print("No search results found.")

        # Get recent summaries
        print(f"\n--- Recent Summaries (last {args.limit}) ---")
        recent_summaries = db.summaries_collection.get(
            limit=args.limit,
            include=["metadatas"]
        )

        if recent_summaries and recent_summaries['metadatas']:
            for i, metadata in enumerate(recent_summaries['metadatas'], 1):
                created_at = metadata.get('created_at', '')
                # Parse ISO timestamp
                try:
                    if created_at:
                        parsed_date = date_parser.parse(created_at)
                        date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_str = "Unknown"
                except Exception:
                    date_str = "Invalid date"

                # Display enhanced metadata including model info
                print(f"{i:2d}. Query: '{metadata.get('original_query', 'N/A')}' "
                      f"Type: {metadata.get('search_type', 'N/A')} "
                      f"Method: {metadata.get('summary_method', 'N/A')} "
                      f"Model: {metadata.get('generator_model', 'N/A')} "
                      f"Date: {date_str}")
        else:
            print("No summaries found.")

    except Exception as e:
        print(f"Error listing recent entries: {e}")


def audit_provenance(args):
    """Show provenance trail for a specific summary."""
    db = get_db()
    try:
        # Search for summaries containing the query
        print(f"Searching for summaries containing: '{args.query}'")

        # Get all summaries and filter by query content
        all_summaries = db.summaries_collection.get(include=["metadatas", "documents"])

        if not all_summaries['metadatas']:
            print("No summaries found in database.")
            return

        matching_summaries = []
        for i, (metadata, document) in enumerate(zip(all_summaries['metadatas'], all_summaries['documents'])):
            if (args.query.lower() in metadata.get('original_query', '').lower() or
                args.query.lower() in document.lower()):
                matching_summaries.append((i, metadata, document))

        if not matching_summaries:
            print(f"No summaries found containing '{args.query}'")
            return

        for idx, (i, metadata, document) in enumerate(matching_summaries, 1):
            print(f"\n=== Summary #{idx} ===")
            print(f"Original Query: {metadata.get('original_query', 'N/A')}")
            print(f"Search Type: {metadata.get('search_type', 'N/A')}")
            print(f"Generated By: {metadata.get('generator_model', 'N/A')}")
            print(f"Method: {metadata.get('summary_method', 'N/A')}")
            print(f"Created: {metadata.get('created_at', 'N/A')}")

            # Show source document IDs
            source_ids = metadata.get('source_result_ids', [])
            print(f"Source Documents: {len(source_ids)} results")

            if source_ids and args.verbose:
                print("\n--- Source Document Details ---")
                # Try to fetch source documents
                try:
                    source_docs = db.results_collection.get(
                        ids=source_ids,
                        include=["metadatas"]
                    )
                    if source_docs['metadatas']:
                        for j, source_meta in enumerate(source_docs['metadatas'], 1):
                            print(f"  {j}. Title: {source_meta.get('title', 'N/A')}")
                            print(f"     URL: {source_meta.get('url', 'N/A')}")
                            print(f"     Retrieved: {source_meta.get('retrieved_at', 'N/A')}")
                            print(f"     API: {source_meta.get('source_api', 'N/A')} ({source_meta.get('source_api_endpoint', 'N/A')})")
                            print()
                    else:
                        print("  (Source documents not found - may have been pruned)")
                except Exception as e:
                    print(f"  Error fetching source documents: {e}")

            if args.show_content:
                print("\n--- Summary Content ---")
                print(document[:500] + "..." if len(document) > 500 else document)
            print("-" * 50)

    except Exception as e:
        print(f"Error auditing provenance: {e}")


def clear_all(args):
    """Clear all cache data (with confirmation)."""
    if not args.force:
        response = input("This will delete ALL cached search data. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return

    db = get_db()
    try:
        # Get counts before deletion
        stats = db.get_stats()
        results_count = stats.get('results_count', 0)
        summaries_count = stats.get('summaries_count', 0)

        # Clear all collections
        print("Clearing all cached data...")

        if results_count > 0:
            # Get all IDs and delete them
            all_results = db.results_collection.get()
            if all_results['ids']:
                db.results_collection.delete(ids=all_results['ids'])
                print(f"Deleted {len(all_results['ids'])} search results.")

        if summaries_count > 0:
            # Get all IDs and delete them
            all_summaries = db.summaries_collection.get()
            if all_summaries['ids']:
                db.summaries_collection.delete(ids=all_summaries['ids'])
                print(f"Deleted {len(all_summaries['ids'])} summaries.")

        print("Cache cleared successfully.")

    except Exception as e:
        print(f"Error clearing cache: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage the SearchWeb Vector DB Cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats command
    parser_stats = subparsers.add_parser("stats", help="Show database statistics.")
    parser_stats.set_defaults(func=show_stats)

    # Prune command
    parser_prune = subparsers.add_parser("prune-by-date", help="Prune cache entries older than N days.")
    parser_prune.add_argument("days", type=int, help="Number of days to keep.")
    parser_prune.set_defaults(func=prune_by_date)

    # List recent command
    parser_list = subparsers.add_parser("list-recent", help="List recent cache entries.")
    parser_list.add_argument("--limit", type=int, default=10, help="Number of recent entries to show (default: 10).")
    parser_list.set_defaults(func=list_recent)

    # Audit provenance command
    parser_audit = subparsers.add_parser("audit", help="Show provenance trail for summaries containing query.")
    parser_audit.add_argument("query", help="Search term to find in summaries or original queries.")
    parser_audit.add_argument("--verbose", "-v", action="store_true", help="Show detailed source document information.")
    parser_audit.add_argument("--show-content", action="store_true", help="Show summary content (first 500 chars).")
    parser_audit.set_defaults(func=audit_provenance)

    # Clear all command
    parser_clear = subparsers.add_parser("clear-all", help="Clear all cached data.")
    parser_clear.add_argument("--force", action="store_true", help="Skip confirmation prompt.")
    parser_clear.set_defaults(func=clear_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()