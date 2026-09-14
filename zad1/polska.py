from polska_class import Polish
def main():
    string, type = str(input("Enter the string and the type of operation (to_polish/to_common/count): ")).split("; ")
    kek = Polish(string,type)
    print(kek.action())


if __name__ == "__main__":
    main()