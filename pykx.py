def convert_kdb_result(kdb_result):
    # Dictionary (handles nested)
    if hasattr(kdb_result, 'keys'):
        result_dict = {}
        for key in kdb_result.keys():
            value = kdb_result[key]
            
            if hasattr(value, 'pd'):              # Table → DataFrame
                result_dict[key] = value.pd()
            elif hasattr(value, 'keys'):          # Nested dict → recurse
                result_dict[key] = convert_kdb_result(value)
            elif hasattr(value, '__len__'):       # List → Series
                result_dict[key] = pd.Series(value.py() if hasattr(value, 'py') else list(value))
            else:                                 # Scalar → Python type
                result_dict[key] = value.py() if hasattr(value, 'py') else value
                
        return result_dict
