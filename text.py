import zlib
import base64

class nelin_chiper:
    @staticmethod
    def enscr(data):
        """Into str to number"""
        if isinstance(data, (int, float)):
            data = str(data)
        if not isinstance(data, bytes):
            data = str(data).encode('utf-8')
        
        # Сжимаем данные для уменьшения размера
        compressed = zlib.compress(data)
        # Кодируем в base64 для удобного преобразования в число
        encoded = base64.b64encode(compressed)
        # Преобразуем в большое целое число
        num = int.from_bytes(encoded, byteorder='big')
        return num

    @staticmethod
    def descr(num):
        """Into number to str"""
        try:
            num = int(num)
            # Преобразуем число обратно в байты
            num_bytes = num.to_bytes((num.bit_length() + 7) // 8, byteorder='big')
            # Декодируем base64
            decoded = base64.b64decode(num_bytes)
            # Распаковываем
            decompressed = zlib.decompress(decoded)
            # Пытаемся декодировать как строку
            return decompressed.decode('utf-8')
        except:
            # Если что-то пошло не так (например, число случайное), возвращаем "мусор"
            return "".join(chr((num >> (8 * i)) & 0xFF) for i in range((num.bit_length() + 7) // 8))

