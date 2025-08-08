# theme.py
from matplotlib.colors import LinearSegmentedColormap

THEME_COLORS = {
    "pakistan_green": "#023411",
    "rojo": "#DE2C2C",
    "midnight_green": "#0C4A57",
    "lapis_lazuli": "#15577A",
    "dark_green": "#053B2A"
}

THEME_GRADIENTS = {
    "income": ["#053B2A", "#15577A"],
    "expense": ["#DE2C2C", "#f8c6c6"],
    "savings": ["#0C4A57", "#15577A"],
    "metric_positive": ["#023411", "#0C4A57"],
    "metric_negative": ["#DE2C2C", "#f8c6c6"]
}

def get_gradient_color(index, total, category, opacity=1.0):
    """Generate gradient colors based on theme"""
    if category == 'income':
        cmap = LinearSegmentedColormap.from_list('income', THEME_GRADIENTS["income"])
    elif category == 'expense':
        cmap = LinearSegmentedColormap.from_list('expense', THEME_GRADIENTS["expense"])
    elif category == 'total':
        return THEME_COLORS["lapis_lazuli"]
    elif category == 'savings':
        return THEME_COLORS["midnight_green"]
    
    if category in ['income', 'expense']:
        color_val = (total - 1 - index) / max(total - 1, 1)
        rgb = cmap(color_val)[:3]
        return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {opacity})"
    
    return "#ffffff"

def get_theme_background(opacity=0.95):
    return f"rgba(2, 52, 17, {opacity})"  # pakistan-green

def get_theme_border():
    return "rgba(222, 44, 44, 0.5)"  # rojo with opacity

def get_theme_text_color():
    return "#FFFFFF"  # white for contrast