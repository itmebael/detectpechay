"""
Test Supabase Connection and Database Setup
Run this script to verify your Supabase connection is working
"""
from config.database import supabase, TABLES, SUPABASE_URL
import traceback

def test_connection():
    print("=" * 60)
    print("Testing Supabase Connection")
    print("=" * 60)
    print(f"URL: {SUPABASE_URL}\n")
    
    # Test 1: Check connection
    print("1. Testing connection...")
    try:
        result = supabase.table('users').select('count').execute()
        print("   [OK] Connection successful!")
    except Exception as e:
        print(f"   [ERROR] Connection failed: {e}")
        return False
    
    # Test 2: Check tables
    print("\n2. Checking tables...")
    for table_name, table_id in TABLES.items():
        try:
            result = supabase.table(table_id).select('count').limit(1).execute()
            print(f"   [OK] Table '{table_id}' exists")
        except Exception as e:
            print(f"   [ERROR] Table '{table_id}' not found or error: {e}")
    
    # Test 3: Check users table
    print("\n3. Checking users table...")
    try:
        result = supabase.table('users').select('id, username, email').limit(5).execute()
        if result.data:
            print(f"   [OK] Found {len(result.data)} user(s):")
            for user in result.data:
                print(f"      - {user.get('username')} ({user.get('email')})")
        else:
            print("   [INFO] No users found (this is OK if you haven't registered yet)")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 4: Check detection_results table
    print("\n4. Checking detection_results table...")
    try:
        result = supabase.table('detection_results').select('count').execute()
        count = len(result.data) if result.data else 0
        print(f"   [OK] Table exists with {count} detection(s)")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 5: Check dataset_images table
    print("\n5. Checking dataset_images table...")
    try:
        result = supabase.table('dataset_images').select('count').execute()
        count = len(result.data) if result.data else 0
        print(f"   [OK] Table exists with {count} dataset image(s)")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    print("\n" + "=" * 60)
    print("Connection Test Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        test_connection()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        traceback.print_exc()

