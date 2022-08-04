from pydantic import BaseModel
from item import Item
from color import Color


class Filename(BaseModel):
    name: str

class Spec:
    @staticmethod
    def item(filename : Filename):
        print(f'{filename}')
        item = Item(filename.name).find_item()
        return item   

    @staticmethod
    def color(filename: Filename):
        print(f'{filename}')
        color = Color(filename.name).discrimination_color()
        return color

    @staticmethod
    def service(filename: Filename):
        print(f'{filename}')


        item = Item(filename).find_item()
        color = Color(filename).discrimination_color()
        return {"item": item, "color":color} 
    