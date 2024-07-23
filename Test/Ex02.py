class Animal(object):
    name: str = '코로링'

    def sound(self):
        print(f'고양이 이름은 {self.name}운다!')


def call(sound:str) -> str:
    print("meow!")
    return sound


animal = Animal()
animal.sound()

print(call(sound = "Meow"))
