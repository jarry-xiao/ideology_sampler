import io

def df_to_postgres(df, table, conn, commit=True):
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='`', header=False, index=False, doublequote=False, escapechar='\\')
    output.seek(0)
    try:
        cur.copy_from(output, table, sep='`', null="")
        if commit:
            conn.commit()
        return True
    except Exception as e:
        print("Encountered error during write", e)
        print("Rolling back.")
        conn.rollback() 
        return False
