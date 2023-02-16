import pyproj


def convert_to_web_mercator(df, col_lat="latitude", col_lon="longitude"):
    """
    Convert (Longitude, Latitude) coordinates (of the whole world) to the web mercator coordinates.
    (The web mercator coordinates work better when computing euclidian distances)

    Inputs:
        df - dataframe with columns [col_lon, col_lat].
        col_lat - string name of the latitude column.
        col_lon - string name of the longitude column.

    Output:
        df - dataframe with new columns ['longitude_merc', 'latitude_merc']
    """
    old_proj = pyproj.Proj("epsg:4326", preserve_units=True)
    new_proj = pyproj.Proj("epsg:3857", preserve_units=True)

    lon = df[col_lon].values
    lat = df[col_lat].values

    x, y = pyproj.transform(old_proj, new_proj, lat, lon)
    df["longitude_merc"] = x
    df["latitude_merc"] = y

    return df
