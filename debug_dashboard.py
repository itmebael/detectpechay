"""
Debug script to test dashboard data fetching
"""
from services.database_service import DatabaseService
from config.database import supabase, TABLES

# Test user IDs from database
test_user_id = '710abbe1-a3e5-4264-a659-673d19ce728d'  # sab user

db_service = DatabaseService()

print("=" * 60)
print("Testing Dashboard Data Fetching")
print("=" * 60)

# Test 1: Check all detections
print("\n1. All detections in database:")
try:
    result = supabase.table(TABLES['detections']).select('*').execute()
    print(f"   Total detections: {len(result.data) if result.data else 0}")
    for det in (result.data or []):
        print(f"   - {det.get('filename')}: user_id={det.get('user_id')}, condition={det.get('condition')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Check stats with user_id
print(f"\n2. Dashboard stats for user_id: {test_user_id}")
try:
    stats = db_service.get_dashboard_stats(test_user_id)
    print(f"   Stats: {stats}")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check detections with user_id
print(f"\n3. Detections for user_id: {test_user_id}")
try:
    results = db_service.get_user_detections(test_user_id, limit=10)
    print(f"   Found {len(results)} detections")
    for r in results:
        print(f"   - {r.get('filename')}: {r.get('condition')}")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check detections without user_id filter
print("\n4. All detections (no user filter):")
try:
    result = supabase.table(TABLES['detections']).select('*').order('timestamp', desc=True).limit(10).execute()
    print(f"   Found {len(result.data) if result.data else 0} detections")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
















