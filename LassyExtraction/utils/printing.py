from typing import Any

SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
SUP = str.maketrans('abcdefghijklmnoprstuvwxyz1', 'ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ¹')
SC = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ1', 'ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ1')


def subscript(x: Any) -> str:
    return str(x).translate(SUB)


def superscript(x: Any) -> str:
    return str(x).translate(SUP)


def smallcaps(x: Any) -> str:
    return str(x).translate(SC)


_box = '□'
_diamond = '◊'


def print_box(x: str) -> str:
    return f'{_box}{superscript(x)}'


def print_diamond(x: str) -> str:
    return f'{_diamond}{superscript(x)}'


def cap(x: Any) -> str:
    # reflects box intro -- box abs
    return f'▴{superscript(x)}'


def cup(x: Any) -> str:
    # reflects box elim -- box app
    return f'▾{superscript(x)}'


def wedge(x: Any) -> str:
    # reflects diamond intro -- diamond app
    return f'▵{superscript(x)}'


def vee(x: Any) -> str:
    # reflects diamond elim -- diamond abs
    return f'▿{superscript(x)}'
