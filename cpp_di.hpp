#ifndef _CPP_DI
#define _CPP_DI

#include <memory>
#include <tuple>
#include <utility>
#include <mutex>
#include <functional>
#include <type_traits>

namespace cpp_di
{
    namespace refl {
        // tag<T, N> generates friend declarations and helps with overload resolution.
        // There are two types: one with the auto return type, which is the way we read types later.
        // The second one is used in the detection of instantiations without which we'd get multiple
        // definitions.
        template <typename T, int N>
        struct tag {
            friend auto loophole(tag<T, N>);
            constexpr friend int cloophole(tag<T, N>);
        };

        // The definitions of friend functions.
        template <typename T, typename U, int N, bool B,
            typename = typename std::enable_if_t<
            !std::is_same_v<
            std::remove_cv_t<std::remove_reference_t<T>>,
            std::remove_cv_t<std::remove_reference_t<U>>>>>
            struct fn_def {
            friend auto loophole(tag<T, N>) { return U{}; }
            constexpr friend int cloophole(tag<T, N>) { return 0; }
        };

        // This specialization is to avoid multiple definition errors.
        template <typename T, typename U, int N> struct fn_def<T, U, N, true> {};

        // This has a templated conversion operator which in turn triggers instantiations.
        // Important point, using sizeof seems to be more reliable. Also default template
        // arguments are "cached" (I think). To fix that I provide a U template parameter to
        // the ins functions which do the detection using constexpr friend functions and SFINAE.
        template <typename T, int N>
        struct c_op {
            template <typename U, int M>
            static auto ins(...) -> int;
            template <typename U, int M, int = cloophole(tag<T, M>{}) >
            static auto ins(int) -> char;

            template <typename U, int = sizeof(fn_def<T, U, N, sizeof(ins<U, N>(0)) == sizeof(char)>)>
            operator U();
        };

        // Here we detect the data type field number. The byproduct is instantiations.
        // Uses list initialization. Won't work for types with user-provided constructors.
        // In C++17 there is std::is_aggregate which can be added later.
        template <typename T, int... Ns>
        constexpr int fields_number(...) { return sizeof...(Ns) - 1; }

        template <typename T, int... Ns>
        constexpr auto fields_number(int) -> decltype(T{ c_op<T, Ns>{}... }, 0) {
            return fields_number<T, Ns..., sizeof...(Ns)>(0);
        }

        // Here is a version of fields_number to handle user-provided ctor.
        // NOTE: It finds the first ctor having the shortest unambigious set
        //       of parameters.
        template <typename T, int... Ns>
        constexpr auto fields_number_ctor(int) -> decltype(T(c_op<T, Ns>{}...), 0) {
            return sizeof...(Ns);
        }

        template <typename T, int... Ns>
        constexpr int fields_number_ctor(...) {
            return fields_number_ctor<T, Ns..., sizeof...(Ns)>(0);
        }

        // This is a helper to turn a ctor into a tuple type.
        // Usage is: refl::as_tuple<data_t>
        template <typename T, typename U> struct loophole_tuple;

        template <typename T, int... Ns>
        struct loophole_tuple<T, std::integer_sequence<int, Ns...>> {
            using type = std::tuple<decltype(loophole(tag<T, Ns>{}))... > ;
        };

        template <typename T>
        using as_tuple =
            typename loophole_tuple<T, std::make_integer_sequence<int, fields_number_ctor<T>(0)>>::type;

    }  // namespace refl

    namespace type_registry
    {
        template<typename>
        struct tag_t
        {
            friend constexpr auto not_detected_in_di_container(tag_t);
        };

        template<typename T>
        struct loophole_t
        {
            friend constexpr auto not_detected_in_di_container(tag_t<T>) { return (T*)nullptr; };
        };

        template<typename T>
        void set_type()
        {
            sizeof(loophole_t<T>);
        }

        template<typename T>
        void check_type()
        {
            using _ = decltype(not_detected_in_di_container(tag_t<T>()));
        }
    }

    namespace internal
    {
        template<typename T> struct is_shared_ptr : std::false_type {};
        template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
    }

    class di
    {
        template<typename CLASS_T, typename ARGUMENT_T>
        static void fail_type()
        {
            static_assert(false, "Constructor for class BASE_T argument type T should be std::shared_ptr.");
        }

        template<class BASE_T, size_t N, class TUPLE_T, class T = typename std::tuple_element<N, TUPLE_T>::type>
        static void construct_tuple_assign(TUPLE_T& tuple)
        {
            if constexpr (internal::is_shared_ptr<T>::value)
            {
                std::get<N>(tuple) = registry<typename T::element_type>::get();
            }
            else
            {
                fail_type<BASE_T, T>();
            }
        }

        template <typename BASE_T, std::size_t From, size_t... indices, typename TUPLE_T>
        static void construct_tuple(TUPLE_T& tuple, std::index_sequence<indices...>)
        {
            (void)std::initializer_list<int>{(construct_tuple_assign<BASE_T, indices, TUPLE_T>(tuple), 0)...};
        }


        template <typename BASE_T, std::size_t From, std::size_t To, typename TUPLE_T>
        static void construct_tuple(TUPLE_T& tuple)
        {
            construct_tuple<BASE_T, From>(tuple, std::make_index_sequence<To - From>());
        }

        template <class T, class TUPLE_T, size_t... indices>
        constexpr static std::shared_ptr<T> make_shared_from_tuple_impl(TUPLE_T&& tuple, std::index_sequence<indices...>)
        {
            return std::make_shared<T>(std::get<indices>(std::forward<TUPLE_T>(tuple))...);
        }

        template<typename T, typename TUPLE_T>
        static std::shared_ptr<T> make_shared_from_tuple(TUPLE_T&& tuple)
        {
            return make_shared_from_tuple_impl<T>(tuple, std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<TUPLE_T>>>{});
        }
    public:


        template<typename T, typename PTR_T = std::shared_ptr<T>, typename CNSTR_T = std::function<PTR_T()>>
        class registry
        {
            inline static std::once_flag flag;
            inline static PTR_T obj;
        public:
            inline static CNSTR_T constructor;

            inline static PTR_T get()
            {
                std::call_once(flag, []() { obj = constructor(); });
                return obj;
            }
        };

        template<typename INTERFACE_T, typename T, std::enable_if_t<(std::is_same_v<INTERFACE_T, T> || std::is_base_of_v<INTERFACE_T, T>) && std::is_default_constructible_v<T>, bool> = true>
        static void add()
        {
            type_registry::set_type<INTERFACE_T>();

            if (registry<INTERFACE_T>::constructor)
            {
                return;
            }

            if constexpr (std::is_same_v<T, INTERFACE_T>)
            {
                registry<T>::constructor = []() { return std::make_shared<T>(); };
            }
            else
            {
                registry<INTERFACE_T>::constructor = []() { return std::static_pointer_cast<INTERFACE_T>(std::make_shared<T>()); };
            }
        }

        template<typename INTERFACE_T, typename T, std::enable_if_t<(std::is_same_v<INTERFACE_T, T> || std::is_base_of_v<INTERFACE_T, T>) && !std::is_default_constructible_v<T>, bool> = true>
        static void add()
        {
            type_registry::set_type<INTERFACE_T>();

            using ctr_type = refl::as_tuple<T>;

            if (registry<INTERFACE_T>::constructor)
            {
                return;
            }

            if constexpr (std::is_same_v<T, INTERFACE_T>)
            {
                registry<INTERFACE_T>::constructor = []()
                {
                    ctr_type args;
                    construct_tuple<T, 0, std::tuple_size_v<ctr_type>>(args);
                    return make_shared_from_tuple<INTERFACE_T>(args);
                };
            }
            else
            {
                registry<INTERFACE_T>::constructor = []()
                {
                    ctr_type args;
                    construct_tuple<T, 0, std::tuple_size_v<ctr_type>>(args);
                    return std::static_pointer_cast<INTERFACE_T>(make_shared_from_tuple<T>(args));
                };
            }
        }

        template<typename T>
        static void add() { type_registry::set_type<T>(); add<T, T>(); }

        template<typename T>
        static std::shared_ptr<T> get()
        {
            type_registry::check_type<T>();
            return registry<T>::get();
        }
    };
}

#endif