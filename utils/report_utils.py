import seaborn as sns
import imgkit
import os


def __file_name_from__(report_name):
    return os.path.join("./report", f"{report_name}-table.png")


def df_to_styled_img(df, report_name, image_width=500, image_height = 280):
    """
    Renders DataFrame into nicely styled html code and convert it into image file
    @param df: DataFrame as input
    @param report_name: desired image file name
    """
    cm = sns.light_palette("seagreen", as_cmap=True)
    styled_table = df.style.background_gradient(cmap=cm)
    html = styled_table.render()

    options = {
        'width': f'{image_width}',
        'height': f'{image_height}',
        'encoding': "UTF-8"
    }
    imgkit.from_string(html, __file_name_from__(report_name), options=options)
