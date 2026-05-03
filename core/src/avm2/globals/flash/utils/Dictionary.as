package flash.utils {
    [Ruffle(InstanceAllocator)]
    public dynamic class Dictionary {
        prototype.toJSON = function(r:String):* {
            return "Dictionary";
        };
        prototype.setPropertyIsEnumerable("toJSON", false);

        public function Dictionary(weakKeys:Boolean = false) {
            // `weakKeys` is accepted but ignored — Ruffle's GC holds
            // dictionary keys strongly. Functional impact is only
            // increased lifetime for keys that the AS3 caller no longer
            // references; semantics for membership/iteration are
            // identical.
        }
    }
}
